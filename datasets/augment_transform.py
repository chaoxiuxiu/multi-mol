from torchvision import transforms

def load_norm_transform(imageSize):#进行图像转换
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.985, 0.982, 0.986],
                                     std=[0.113, 0.119, 0.112])
    img_tra = transforms.Compose([
       transforms.Resize((imageSize,imageSize)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomGrayscale(p=0.2),
       transforms.RandomRotation(degrees=360),
       transforms.ToTensor(),
       normalize,
    ])
    return img_tra

def load_test_transform(imageSize):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.985, 0.982, 0.986],
                                     std=[0.113, 0.119, 0.112])
    img_tra = transforms.Compose([
       transforms.Resize((imageSize,imageSize)),
       # transforms.RandomGrayscale(p=0.2),
       transforms.ToTensor(),
       normalize,
    ])
    return img_tra