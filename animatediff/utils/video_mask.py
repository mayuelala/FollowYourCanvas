import random
import torch
import torch.nn.functional as F

def video_mask(pixel_values, pixel_pad=None):
    [_, _, _, h, w] = pixel_values.shape

    min_rect_w = int(w * 1 / 4)
    min_rect_h = int(h * 1 / 4)

    max_rect_w = int(w * 3 / 4)
    max_rect_h = int(h * 3 / 4)

    # Generate mask templete
    mask = torch.ones_like(pixel_values)[:, :, 0:1, :, :]
    mask.to(pixel_values.device)
    
    mask_choice = ['horizontal outpaint', 'vertical outpaint', 'float outpaint']

    if pixel_pad is None:
        n = random.uniform(0, 1)
        if n < 0.4:
            mask_use = ['horizontal outpaint']
        elif 0.4 <= n < 0.8:
            mask_use = ['vertical outpaint']
        else:
            mask_use = ['float outpaint']
    else:
        mask_use = 'specific'
        for idx in range(len(pixel_pad)):
            [pad_up, pad_down, pad_left, pad_right] = pixel_pad[idx].tolist()
            if pad_right == 0 and pad_left==0:
                mask[idx:(idx + 1), :, :, pad_up:-pad_down, :] = 0
            elif pad_up == 0 and pad_down==0:
                mask[idx:(idx + 1), :, :, :, pad_left:-pad_right] = 0
            else:
                mask[idx:(idx + 1), :, :, pad_up:-pad_down, pad_left:-pad_right] = 0

    if 'horizontal outpaint' in mask_use:
        rect_w = random.randint(min_rect_w, max_rect_w)
        
        rect_start_w = random.randint(0, int(w - rect_w))
        rect_end_w = int(rect_start_w + rect_w)
        
        mask[:, :, :, :, rect_start_w:rect_end_w] = 0

    elif 'vertical outpaint' in mask_use:
        rect_h = random.randint(min_rect_h, max_rect_h)

        rect_start_h = random.randint(0, int(h - rect_h))
        rect_end_h = int(rect_start_h + rect_h)

        mask[:, :, :, rect_start_h:rect_end_h, :] = 0

    elif 'float outpaint' in mask_use:
        rect_w = random.randint(min_rect_w, max_rect_w)
        
        rect_start_w = random.randint(0, int(w - rect_w))
        rect_end_w = int(rect_start_w + rect_w)

        rect_h = random.randint(min_rect_h, max_rect_h)

        rect_start_h = random.randint(0, int(h - rect_h))
        rect_end_h = int(rect_start_h + rect_h)
        
        mask[:, :, :, rect_start_h:rect_end_h, rect_start_w:rect_end_w] = 0

    return mask


def get_anchor_target(pixel_values, anchor_target_sampling):
    target_size = anchor_target_sampling['target_size']
    anchor_size = anchor_target_sampling['anchor_size']
    dynamic_anchor = anchor_target_sampling['dynamic_anchor_size']
    if len(pixel_values.shape) == 4:
        pixel_values = pixel_values.unsqueeze(0)
    [b, f, c, h, w] = pixel_values.shape
    mask = torch.ones([b, f, 1, target_size[0], target_size[1]], device=pixel_values.device)

    max_target_h = int(h - target_size[0])
    max_target_w = int(w - target_size[1])

    if dynamic_anchor:
        anchor_size = [int(random.uniform(anchor_size[0], anchor_size[1])), int(random.uniform(anchor_size[2], anchor_size[3]))]
    if 'overlap_ratio' in list(anchor_target_sampling.keys()):
        # sample overlap ratio
        [min_overlap_h, max_overlap_h, min_overlap_w, max_overlap_w] = anchor_target_sampling['overlap_ratio']
        overlap_h = random.uniform(min_overlap_h, max_overlap_h)
        overlap_w = random.uniform(min_overlap_w, max_overlap_w)
        offset_h = int((1-overlap_h) * target_size[0])
        offset_w = int((1-overlap_w) * target_size[1])
        required_video_size = [int((offset_h + anchor_size[0]/2)*2)+16, int((offset_w + anchor_size[1]/2)*2)+16]
        if required_video_size[0]*16/9 > required_video_size[1]:
            scale_factor = required_video_size[0] / h
            new_h = required_video_size[0]
            new_w = int(w * scale_factor)
        else:
            scale_factor = required_video_size[1] / w
            new_h = int(h * scale_factor)
            new_w = required_video_size[1]
        resized_pixel_values = F.interpolate(pixel_values.view(b * f, c, h, w), size=(new_h, new_w), mode='bilinear', align_corners=False)
        resized_pixel_values = resized_pixel_values.view(b, f, c, new_h, new_w)
        # offset left/right

        anchor_center_h = int(new_h/2)
        anchor_center_w = int(new_w/2)
        target_center_h = anchor_center_h + offset_h + int(anchor_size[0]/2) - int(target_size[0]/2) if random.random() > 0.5 else anchor_center_h - offset_h - int(anchor_size[0]/2) + int(target_size[0]/2)
        target_center_w = anchor_center_w + offset_w + int(anchor_size[1]/2) - int(target_size[1]/2) if random.random() > 0.5 else anchor_center_w - offset_w - int(anchor_size[1]/2) + int(target_size[1]/2)
        anchor_h = anchor_center_h - int(anchor_size[0]/2)
        anchor_w = anchor_center_w - int(anchor_size[1]/2)
        target_h = target_center_h - int(target_size[0]/2)
        target_w = target_center_w - int(target_size[1]/2)
        
        pixel_values = resized_pixel_values

    else:
        # sample target position
        target_h = random.randint(0, max_target_h)
        target_w = random.randint(0, max_target_w)

        anchor_h = int(h/2 - anchor_size[0]/2)
        anchor_w = int(w/2 - anchor_size[1]/2)

    known_top = max(anchor_h, target_h) - target_h
    known_left = max(anchor_w, target_w) - target_w
    known_down = max(min(anchor_h + anchor_size[0], target_h + target_size[0]) - target_h, 0)
    known_right = max(min(anchor_w + anchor_size[1], target_w + target_size[1]) - target_w, 0)

    target_pixels_values = pixel_values[:,:,:,target_h:target_h+target_size[0], target_w:target_w+target_size[1]]
    anchor_pixels_values = pixel_values[:,:,:,anchor_h:anchor_h+anchor_size[0], anchor_w:anchor_w+anchor_size[1]]

    if target_pixels_values.shape[3] != target_size[0] or target_pixels_values.shape[4] != target_size[1]:
        import pdb; pdb.set_trace()
    
    # currently the code needs the known areas to be existed
    if 'overlap_ratio' in list(anchor_target_sampling.keys()):
        if abs(offset_h) < anchor_size[0] and abs(offset_w) < anchor_size[1]:
            if not known_top<=known_down and known_left<=known_right:
                import pdb; pdb.set_trace()
            mask[:, :, :, known_top:known_down, known_left:known_right] = 0
        else:
            mask[:, :, :, known_top:known_down, known_left:known_right] = 0
    # same sampling for each batch
    relative_position = torch.tensor([(target_h + int(target_size[0]/2) - (anchor_h + int(anchor_size[0]/2))), ((target_w + int(target_size[1]/2)) - (anchor_w+ int(anchor_size[1]/2))), anchor_size[0], anchor_size[1], target_size[0], target_size[1]], device=pixel_values.device)
    relative_position = relative_position.unsqueeze(0).repeat(b, 1)

    return anchor_pixels_values, target_pixels_values, mask, relative_position