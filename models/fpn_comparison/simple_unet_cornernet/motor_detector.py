from .model_defs import cascade_fpn, parallel_fpn, simple_unet

import torch
import monai
import gc
import monai.inferers#we do need this but why? im nto sure maybe some weird python quirk
PRESETS = {
    "simple_unet": {
        "4m": {"base_features":5, "growth_rate": 2} ,
        "10m":  {"base_features":8, "growth_rate": 2}
    },
    "parallel_fpn": {
        "4m" : {"base_features":4, "growth_rate" : 2.3, "downchanneling_factor" : 2 },
        "10m" : {"base_features":6, "growth_rate" : 2.3, "downchanneling_factor" : 2 }
    },
    "cascade_fpn": {
        "4m" : {"base_features":4, "growth_rate" : 2.3, "downchanneling_factor" : 2 }
    }
}

def _load_preset(model_name, model_size, dropout_p, drop_path_p):
    preset = None
    if model_name == 'simple_unet':
        preset = PRESETS['simple_unet'].get(model_size, None)
        if preset is None:
            raise Exception(f"model: {model_name} does not have preset: {model_size}\nExisting Presets: {PRESETS['simple_unet']}")
        model = simple_unet.SimpleUNET(**preset, dropout_p=dropout_p, drop_path_p=drop_path_p)
    elif model_name == 'parallel_fpn':
        preset = PRESETS['parallel_fpn'].get(model_size, None)
        if preset is None:
            raise Exception(f"model: {model_name} does not have preset: {model_size}\nExisting Presets: {PRESETS['parallel_fpn']}")
        model = parallel_fpn.ParallelFPN(**preset, dropout_p=dropout_p, drop_path_p=drop_path_p)
    elif model_name == 'cascade_fpn':
        preset = PRESETS['cascade_fpn'].get(model_size, None)
        if preset is None:
            raise Exception(f"model: {model_name} does not have preset: {model_size}\nExisting Presets: {PRESETS['cascade_fpn']}")
        model = cascade_fpn.CascadeFPN(**preset, dropout_p=dropout_p, drop_path_p=drop_path_p)
    else:
        raise Exception(f"Unknown model: {model_name}\nAvailable models: {list(PRESETS.keys())}")
    return model, preset
class MotorDetector(torch.nn.Module):

    def __init__(self, model_name: str, model_size: str, dropout_p: float, drop_path_p: float):
        super().__init__()
        self.model_name = model_name
        self.model_size = model_size
        self.model, _ = _load_preset(model_name=model_name, model_size=model_size, dropout_p=dropout_p, drop_path_p=drop_path_p)

    def save_checkpoint(self, path, optimizer):
        state = {
            "weights": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_name": self.model_name,
            "model_size": self.model_size,
        }
        torch.save(obj=state, f=path)

    @classmethod
    def load_checkpoint(cls, path, dropout_p=0, drop_path_p=0):
        """Load a model from checkpoint.

        Args:
            path: Path to checkpoint file
            dropout_p: Dropout probability (defaults to 0 for inference)
            drop_path_p: Drop path probability (defaults to 0 for inference)

        Returns:
            (MotorDetector instance, optimizer_state dict)
        """
        state = torch.load(path)
        detector = cls(
            model_name=state['model_name'],
            model_size=state['model_size'],
            dropout_p=dropout_p,
            drop_path_p=drop_path_p
        )
        detector.model.load_state_dict(state['weights'])
        return detector, state['optimizer_state']
    
    def forward(self, x):
        return self.model(x)
    
    def print_params(self):
        self.model.print_params()
        
    @torch.inference_mode()
    def inference(self, tomo_tensor, batch_size, patch_size, overlap, device = torch.device('cuda'), tqdm_progress:bool = False,dtype = torch.float32, sigma_scale = 1/8, mode = 'gaussian'):
        # sigmoid_model = MotorIdentifierWithSigmoid(self)
        inferer = monai.inferers.inferer.SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=batch_size, overlap=overlap, 
            mode=mode, sigma_scale=sigma_scale, device=device, 
            progress=tqdm_progress, buffer_dim=0
        )
        
        with torch.amp.autocast(dtype = dtype, device_type="cuda"):
            results = inferer(inputs=tomo_tensor, network=self.model)
        
        del inferer
        torch.cuda.empty_cache()
        gc.collect()
        
        return torch.sigmoid(results)