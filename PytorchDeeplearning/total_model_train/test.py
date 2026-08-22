import torch
from torchvision import transforms
from PIL import Image
device = torch.device('cuda')
image_path = "./airplane.png"
image = Image.open(image_path).convert('RGB')

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
image = trans(image)
image = image.to(device)
print(image.shape)

# print(image)
from model import CIFARmodel
model = CIFARmodel()
model.load_state_dict(torch.load("./model/model_99_gpu.pth"))
model.to(device)
# print(model)
image = torch.reshape(image, (1, 3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(torch.argmax(output, dim=1).item())