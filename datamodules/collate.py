import torch

def custom_collate(batch):

    ret = {
            "image":                torch.stack([x['image'] for x in batch]),
            "boxes":                [torch.tensor(x['boxes'], dtype=torch.float32) for x in batch],
            "exemplars":            [torch.tensor(x['exemplars'], dtype=torch.float32) for x in batch],

            "img_name":             [x['img_name'] for x in batch],
            "img_url":              [x['img_url'] for x in batch],
            "img_id":               [x['img_id'] for x in batch],
            "img_size":             [x['img_size'] for x in batch],
            "orig_boxes":           [x['orig_boxes'] for x in batch],
            "orig_exemplars":       [x['orig_exemplars'] for x in batch],
    }

    for key in batch[0].keys():
        if key not in ret:
            ret[key] = [x[key] for x in batch]
    
    return ret