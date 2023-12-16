# Convolution blocks after reshaping 

# deconformable conv2d 
```markdown
from dcn import DeformableConv2d

class Model(nn.Module):
    ...
    self.conv = DeformableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
    ...
```
