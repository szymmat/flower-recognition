import torch, argparse, json
from torchvision import models, transforms
from torch import nn
from collections import OrderedDict
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.efficientnet_b0(pretrained=True, progress=True) if checkpoint["arch"] == "efficientnet" else models.densenet121(pretrained=True, progress=True)
    start_units = 1280 if checkpoint["arch"] == "efficientnet" else 1024
    for param in model.parameters():
        param.requires_grad = False

    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(start_units, checkpoint["hidden_units"])),
                            ('relu', nn.ReLU()),
                            ('dropout' ,nn.Dropout(0.2)),
                            ('fc2', nn.Linear(checkpoint["hidden_units"], 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.classifier = classifier
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    return model

def predict(args):
    model = load_checkpoint(args.checkpoint)
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    image = test_transforms(Image.open(args.path_to_image)).float().unsqueeze_(0)
    device = torch.device("cuda" if args.gpu == True else "cpu")
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f, strict=False)
    model.eval()
    model.to(device)
    with torch.no_grad():    
        logps = model.forward(image)
        ps = torch.exp(logps)
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        probs, classes = ps.topk(args.top_k, dim=1)
        int_classes = [int(c) for c in classes[0]]
        return probs, [cat_to_name[inv_map[c]] for c in int_classes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', help="Path to image")
    parser.add_argument('checkpoint', help="Path to network checkpoint", default="./checkpoint.pth")
    parser.add_argument('--top_k', help="Number of top probabilities", default=3, type=int)
    parser.add_argument('--category_names', help="Path to JSON file with mapping of categories to real names", default="./cat_to_name.json")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    print(predict(parser.parse_args()))