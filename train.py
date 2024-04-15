import argparse, torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import OrderedDict

def train(args):
    device = torch.device("cuda" if args.gpu == True else "cpu")
    trainloader, testloader, validloader, class_to_idx = prepare_data(args.data_dir)
    if (args.arch == "densenet"):
        model = models.densenet121(pretrained=True, progress=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, args.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout' ,nn.Dropout(0.2)),
                            ('fc2', nn.Linear(args.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        model.classifier = classifier
    elif (args.arch == "efficientnet"):
        model = models.efficientnet_b0(pretrained=True, progress=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1280, args.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout' ,nn.Dropout(0.2)),
                            ('fc2', nn.Linear(args.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
        model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 60
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            print(steps)

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = class_to_idx
    checkpoint = {"arch": args.arch, "hidden_units": args.hidden_units, "state_dict": model.state_dict(), "class_to_idx": class_to_idx}
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
    

def prepare_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32)
    validloader = DataLoader(valid_data, batch_size=32)
    return trainloader, testloader, validloader, train_data.class_to_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="Directory with flower data")
    parser.add_argument('--save_dir', help="Directory to save checkpoints", default=".")
    # The number of different available architectures in PyTorch is so big that I don't want to support all of them
    parser.add_argument('--arch', help="Architecture to use (densenet or efficientnet)", default="efficientnet", choices=["densenet", "efficientnet"])
    parser.add_argument('--learning_rate', help="Learning rate", default=0.003, type=float)
    parser.add_argument('--hidden_units', help="Number of hidden units", default=500, type=int)
    parser.add_argument('--epochs', help="Number of epochs", default=1, type=int)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    train(parser.parse_args())