import os
from dataset.communal.read import read_image, read_zind
from dataset.communal.base_dataset import BaseDataset
from preprocessing.filter import filter_center, filter_boundary, filter_self_intersection



class ZindDataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, aug=None,is_simple=True, is_ceiling_flat=False, vp_align=False):
        super().__init__(mode, shape, aug)
        self.root_dir = root_dir
        self.vp_align = vp_align

        data_dir = os.path.join(root_dir)
        


        pano_list = read_zind(partition_path=os.path.join(data_dir, f"zind_partition.json"),
                              simplicity_path=os.path.join(data_dir, f"room_shape_simplicity_labels.json"),
                              data_dir=data_dir, mode=mode, is_simple=is_simple, is_ceiling_flat=is_ceiling_flat)

        self.data = []
        invalid_num = 0
        for pano in pano_list:
            if not os.path.exists(pano['img_path']):
                invalid_num += 1
                continue

            if not filter_center(pano['corners']):
                invalid_num += 1
                continue

            if not filter_boundary(pano['corners']):
                invalid_num += 1
                continue

            if not filter_self_intersection(pano['corners']):
                invalid_num += 1
                continue

            if 'TWV' not in pano:
                invalid_num += 1
                continue

            if len(pano['TWV']) != 256:
                invalid_num += 1
                continue
            
            self.data.append(pano)

        print(
            f"Build dataset mode: {self.mode} valid: {len(self.data)} invalid: {invalid_num}")

    def __getitem__(self, idx):
        pano = self.data[idx]
        rgb_path = pano['img_path'].replace('panos','vp_aligned_panos')
        label = pano
        image = read_image(rgb_path, self.shape) # (512, 1024, 3)

        # if self.vp_align:
        #     rotation = calc_rotation(corners=label['vp_aligned_corners'])
        #     shift = math.modf(rotation / (2 * np.pi) + 1)[0]
        #     image = np.roll(image, round(shift * self.shape[1]), axis=1)
        #     depth_img = np.roll(depth_img, round(shift * self.shape[1]), axis=1)
        #     normal_img = np.roll(normal_img, round(shift * self.shape[1]), axis=1)

        #     label['trivialWalls'] = np.roll(label['trivialWalls'], round(shift * 256))
        #     label['corners'][:, 0] = np.modf(label['corners'][:, 0] + shift)[0]

        output = self.process_data(label, image)
        return output
