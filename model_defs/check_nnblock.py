import torch
import torch.nn as nn
import nnblock  # Your existing nnblock module

class YourModel(nn.Module):
    def __init__(self, norm_type='gn', num_classes=1):  # Added num_classes parameter
        super().__init__()
        
        FOC = 96  # Reduced from 128

        # Encoder (Downsampling path)
        self.encoder = nn.Sequential(
            #c16, 64^3
            nn.Conv3d(1, FOC//8, kernel_size=5, stride=1, padding = 2),
            nnblock.PreActResBlock3d(FOC//8, FOC//8, norm_type=norm_type),
            #c32, 64^3
            nnblock.PreActResBlock3d(FOC//8, FOC//4, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//4, FOC//4, norm_type=norm_type),
            #c64, 16^3
            nnblock.PreActResBlock3d(FOC//4, FOC//2, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//2, FOC//2, norm_type=norm_type),
            #c128, 8^3
            nnblock.PreActResBlock3d(FOC//2, FOC, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
            nn.Dropout3d(p=0.1, inplace=True)
        )
        
        # Decoder (Upsampling path)
        self.decoder = nn.Sequential(
            #64, 16^3
            nnblock.UpsamplePreActResBlock3d(FOC, FOC//2, stride=2, kernel_size=3, norm_type=norm_type),
            
            #32, 32^3
            nnblock.UpsamplePreActResBlock3d(FOC//2, FOC//4, stride=2, kernel_size=3, norm_type=norm_type),
            
            #16, 64^3
            nnblock.UpsamplePreActResBlock3d(FOC//4, FOC//8, stride=2, kernel_size=3, norm_type=norm_type),
        )

        # Classification head - Fixed the incomplete Conv3d layer
        self.classification_head = nn.Sequential(
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=FOC//8),
            nn.SiLU(inplace=True),
            nn.Conv3d(FOC//8, num_classes, stride=1, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        return self.classification_head(self.decoder(self.encoder(x)))

def test_model_shapes():
    model = YourModel()
    model.eval()
    
    # Test input - assuming 64x64x64 input volume
    input_tensor = torch.randn(1, 1, 96, 96, 96)
    print(f"Input shape: {input_tensor.shape}")
    
    # Track shapes through encoder
    x = input_tensor
    print("\n=== ENCODER SHAPES ===")
    
    encoder_layers = [
        "Conv3d(1->12, k=5, s=1)",
        "PreActResBlock3d(12->12)",
        "PreActResBlock3d(12->24, s=2)",
        "PreActResBlock3d(24->24)",
        "PreActResBlock3d(24->48, s=2)",
        "PreActResBlock3d(48->48)",
        "PreActResBlock3d(48->96, s=2)",
        "PreActResBlock3d(96->96)",
        "PreActResBlock3d(96->96)",
        "Dropout3d"
    ]
    
    for i, layer in enumerate(model.encoder):
        x = layer(x)
        print(f"After {encoder_layers[i]}: {x.shape}")
    
    encoded = x
    print(f"\nEncoded feature shape: {encoded.shape}")
    
    # Track shapes through decoder
    print("\n=== DECODER SHAPES ===")
    
    decoder_layers = [
        "UpsamplePreActResBlock3d(96->48, s=2)",
        "UpsamplePreActResBlock3d(48->24, s=2)", 
        "UpsamplePreActResBlock3d(24->12, s=2)"
    ]
    
    for i, layer in enumerate(model.decoder):
        x = layer(x)
        print(f"After {decoder_layers[i]}: {x.shape}")
    
    decoded = x
    print(f"\nDecoded feature shape: {decoded.shape}")
    
    # Track shapes through classification head
    print("\n=== CLASSIFICATION HEAD SHAPES ===")
    
    head_layers = [
        "Normalization",
        "SiLU",
        "Conv3d(12->1, k=3, s=1, p=1)"
    ]
    
    for i, layer in enumerate(model.classification_head):
        x = layer(x)
        print(f"After {head_layers[i]}: {x.shape}")
    
    final_output = x
    print(f"\nFinal output shape: {final_output.shape}")
    
    # Full forward pass
    print("\n=== FULL FORWARD PASS ===")
    with torch.no_grad():
        full_output = model(input_tensor)
        print(f"Model output shape: {full_output.shape}")
    
    return model

if __name__ == "__main__":
    model = test_model_shapes()