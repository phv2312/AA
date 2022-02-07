import cv2
import shutil
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams, softmax_rgb_blend
from pytorch3d.renderer.mesh.rasterizer import Fragments, rasterize_meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
)


class GeekShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class GeekMeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes which have already been transformed.
    """

    def __init__(self, raster_settings=None):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_screen: a Meshes object representing a batch of meshes with
                          coordinates already projected.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """

        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        verts_ndc = meshes_world._verts_list[0].unsqueeze(0)

        h, w = self.raster_settings.image_size
        verts_ndc[:,:,0] *= w / h # for non-square coordinate system
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        verts_ndc[..., 2] = 1.1 # by default, the z_near is 1.

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_ndc,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=False,
        )
        return Fragments(
            pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
        )


class PytorchRenderer:
    def __init__(self, use_gpu=True):
        # Setup
        if torch.cuda.is_available() and use_gpu:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        self.device = device

    def render_w_texture(self, obj_path, texture_path):
        #
        tmp_output_dir = "temporary_pt3d_mesh_rendering"
        os.makedirs(tmp_output_dir, exist_ok=True)

        #
        tmp_obj_paths, _, _ = to_pt3d_format([obj_path], texture_path, tmp_output_dir)
        tmp_obj_path = tmp_obj_paths[0]
        image = Image.open(texture_path)
        h, w = image.height, image.width

        #
        textured_image = self.render(tmp_obj_path, (w, h))

        # clean up
        shutil.rmtree(tmp_output_dir)

        # return
        return textured_image

    def render(self, obj_path, img_size):
        #
        w, h = img_size

        # Load obj file
        mesh = load_objs_as_meshes([obj_path], device=self.device)

        raster_settings = RasterizationSettings(
            image_size=(h, w),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=GeekMeshRasterizer(raster_settings=raster_settings),
            shader=GeekShader()
        )

        # do rendering
        images = renderer(mesh)

        # post-process
        image = images[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.flip(image, 1) # horizontally
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image


def normalize_line(line:str):
    new_line = ""
    if line.startswith('v '):
        coords = [float(s) for s in line.split(' ')[1:]]

        # # convert from range [0, 1] -> [-1., 1.]
        # for i in range(2):
        #     coords[i] = 2 * (coords[i] - 0.5)
        # coords[1] = - coords[1]

        coords = [str(coord) for coord in coords]
        new_line = "v " + " ".join(coords)
        return new_line
    else:
        new_line = line

    return new_line


def to_pt3d_format(obj_file_paths, texture_file_path, output_dir):
    """
    Convert the original file path (multiple) to pytorch 3d format

    @param obj_file_paths:
    @param texture_file_path:
    @param output_dir
    @return: [pt3d_obj_file_paths, mtl_file_path, texture_file_path]
    """

    #
    result_obj_file_paths = []

    # construct .mtl
    texture_basename = os.path.basename(texture_file_path)
    result_texture_file_path = os.path.join(output_dir, texture_basename)
    shutil.copy(texture_file_path, result_texture_file_path)

    #
    mtl_lines = [
        "newmtl material_1",
        "map_Kd %s" % texture_basename,
        "",
        "# Test colors",
        "",
        "Ka 1.000 1.000 1.000  # white",
        "Kd 1.000 1.000 1.000  # white",
        "Ks 0.000 0.000 0.000  # black",
        "Ns 10.0"
    ]

    result_mtl_file_path = os.path.join(output_dir, 'simple_material.mtl')
    mtl_writer = open(result_mtl_file_path, 'w')
    mtl_writer.write('\n'.join(mtl_lines))
    mtl_writer.close()

    #
    # read -> re-write -> save the obj content

    for obj_file_path in obj_file_paths:
        # read
        lines = open(obj_file_path, 'r').readlines()
        lines = [l.strip() for l in lines]
        lines = [normalize_line(line) for line in lines]

        # re-write
        ids_with_vt = [i for i,l in enumerate(lines) if l.startswith('vt')]
        before_face_idx = ids_with_vt[-1] + 1

        lines.insert(before_face_idx, "usemtl material_1")
        lines.insert(0, "mtllib simple_material.mtl")

        # save
        obj_basename = os.path.basename(obj_file_path)
        obj_out_path = os.path.join(output_dir, obj_basename)

        obj_writer = open(obj_out_path, 'w')
        obj_writer.write("\n".join(lines))
        obj_writer.close()

        result_obj_file_paths += [obj_out_path]

    return result_obj_file_paths, result_texture_file_path, result_mtl_file_path


if __name__ == '__main__':
    obj_file_path = "./../deformed_mesh.obj"
    tex_file_path = "./../data/image_alok.png"

    pt_renderer = PytorchRenderer(use_gpu=False)
    image = pt_renderer.render_w_texture(obj_file_path, tex_file_path)

    print('shape:', image.shape)
    image = image[::-1, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.show()

    # cv2.imwrite('pt3d_format/output_rendered.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))