import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Subset

class SqueezeAndForward(torch.nn.Module):
    """
    A wrapper that squeezes the input tensor from (B, 1, C, T) to (B, C, T)
    before passing it to the underlying model. This is necessary for Captum's
    internal forward passes to work correctly with the EEGNeX model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.squeeze(1))

def compute_and_plot_attributions(
    model,
    dataset,
    test_indices,
    device,
    output_dir,
    fold_num,
    run_dir_name,
    test_subjects,
    ch_names: list,
    times_ms: np.ndarray,
):
    """
    Computes and saves the average attribution map and summary for a model on a given test set.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Wrap the model to handle the input shape mismatch inside Captum
    model_wrapper = SqueezeAndForward(model).to(device)
    ig = IntegratedGradients(model_wrapper)
    
    # Create a DataLoader for only the specified test indices
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
    
    all_attributions = []
    total_correct = 0
    
    # We need to calculate gradients for Integrated Gradients, so no torch.no_grad() here
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Enable gradients for attribution calculation on the input tensor
        inputs.requires_grad_()
        
        # Use the wrapper for the forward pass to get predictions
        outputs = model_wrapper(inputs)
        predicted_classes = torch.argmax(outputs, dim=1)
        
        # Filter for correctly classified samples
        correct_mask = (predicted_classes == labels)
        if not torch.any(correct_mask):
            continue
            
        correct_inputs = inputs[correct_mask]
        correct_labels = labels[correct_mask]
        
        total_correct += correct_inputs.size(0)
        
        # Compute attributions for the correct predictions. 
        # Pass the original 4D tensor; the wrapper will handle the squeeze.
        attributions = ig.attribute(
            correct_inputs,
            target=correct_labels,
            internal_batch_size=correct_inputs.size(0)
        )
        all_attributions.append(attributions.cpu().detach().numpy())
        
    if not all_attributions:
        print(" -> No correct predictions found in test set. Skipping XAI plot.")
        return
        
    # Average the attributions across all correct samples
    avg_attributions = np.mean(np.concatenate(all_attributions, axis=0), axis=0)
    avg_attributions = np.squeeze(avg_attributions)

    # --- Save Outputs ---
    # 1. Raw Attribution Data (.npy)
    npy_path = output_dir / f"fold_{fold_num:02d}_xai_attributions.npy"
    np.save(npy_path, avg_attributions)

    # 2. Heatmap Plot (.png)
    png_path = output_dir / f"fold_{fold_num:02d}_xai_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(avg_attributions, cmap='inferno', aspect='auto', interpolation='nearest')
    ax.set_title(f'Mean Feature Attributions (Fold {fold_num})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('EEG Channels')
    
    # Add meaningful labels
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6) # Use smaller font for readability
    xtick_indices = np.linspace(0, len(times_ms) - 1, num=10, dtype=int)
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xtick_indices])

    fig.colorbar(im, ax=ax, label='Attribution Score')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    # 3. Metadata Summary (.json)
    summary_path = output_dir / f"fold_{fold_num:02d}_xai_summary.json"
    summary_data = {
        "run_dir": run_dir_name,
        "fold": fold_num,
        "test_subjects": test_subjects,
        "xai_method": "Integrated Gradients",
        "num_correct_trials_explained": total_correct,
        "attribution_map_shape": list(avg_attributions.shape),
        "attribution_data_file": npy_path.name,
        "heatmap_image_file": png_path.name
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f" -> XAI outputs for fold {fold_num} saved to {output_dir.name}/")
