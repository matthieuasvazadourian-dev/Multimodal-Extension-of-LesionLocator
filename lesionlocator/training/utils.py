import torch

def training_collate_fn(batch):
    """
    Efficiently collate a list of dicts into a dict of stacked tensors/lists.
    Assumes all dicts have the same keys.
    """
    if len(batch) == 1:
        return batch[0]

    # Stack tensors for each key; keep lists for non-tensor values
    collated = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # Stack if all values are tensors and have the same shape
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values, dim=0)
            except Exception as e:
                print(f"Stacking failed for key '{key}': {e}")
                collated[key] = values  # fallback to list
        else:
            collated[key] = values
    return collated


def tracking_collate_fn(batch):
    """
    Custom collate function for tracking training samples.
    Handles baseline data, follow-up data, baseline prompt, and target.
    """
    if len(batch) == 1:
        return batch[0]
    
    # Stack all batch items into proper tensors
    batch_baseline = []
    batch_followup = []
    batch_prompts = []
    batch_targets = []
    batch_properties = []
    batch_lesion_ids = []
    batch_filenames = []

    for item in batch:
        batch_baseline.append(item['baseline_data'])
        batch_followup.append(item['followup_data'])
        batch_prompts.append(item['baseline_prompt'])
        batch_targets.append(item['target'])
        batch_properties.append(item['properties'])
        batch_lesion_ids.append(item['lesion_id'])
        batch_filenames.append(item['filename'])
    
    # Stack tensors - all should have same dimensions due to preprocessing
    try:
        stacked_baseline = torch.stack(batch_baseline, dim=0)       # [B, C, H, W, D]
        stacked_followup = torch.stack(batch_followup, dim=0)       # [B, C, H, W, D]
        stacked_prompts = torch.stack(batch_prompts, dim=0)         # [B, 1, H, W, D]
        stacked_targets = torch.stack(batch_targets, dim=0)         # [B, H, W, D]
        
        return {
            'baseline_data': stacked_baseline,
            'followup_data': stacked_followup,
            'baseline_prompt': stacked_prompts,
            'target': stacked_targets,
            'properties': batch_properties,
            'lesion_id': batch_lesion_ids,
            'filename': batch_filenames
        }

    except RuntimeError as e:
        # Print shapes for debugging
        print(f"Tracking batch stacking failed: {e}")
        print(f"Baseline shapes: {[d.shape for d in batch_baseline]}")
        print(f"Follow-up shapes: {[d.shape for d in batch_followup]}")
        print(f"Prompt shapes: {[p.shape for p in batch_prompts]}")
        print(f"Target shapes: {[t.shape for t in batch_targets]}")
        # Fallback: process as batch_size=1
        print(f"Warning: Could not stack tracking batch, falling back to single sample processing")
        return batch[0]
