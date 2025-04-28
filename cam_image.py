import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from mymodels.classical_model import *
import os
from torchvision import transforms


def generate_gradcam(model, image_tensor, target_layer, device):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output if not isinstance(output, tuple) else output[0])

    model.to(device)
    image_tensor = image_tensor.to(device)

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)

    try:
        output, _ = model(image_tensor)
    except ValueError:
        output = model(image_tensor)
        if isinstance(output, dict):
            output = output['logits']

    target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()

    handle1.remove()
    handle2.remove()

    grad = gradients[0].detach()
    act = activations[0].detach()

    pooled_grad = torch.mean(grad, dim=[0, 2, 3], keepdim=True) if grad.ndimension() == 4 else torch.mean(grad, dim=[0, 2], keepdim=True)
    cam = torch.sum(pooled_grad * act, dim=1).squeeze()
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.cpu().numpy()


def overlay_heatmap(image_path, cam):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


def create_white_background_image(image, height, width, padding=20):
    image_resized = cv2.resize(image, (width - 2 * padding, height - 2 * padding))
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    y_offset = (height - image_resized.shape[0]) // 2
    x_offset = (width - image_resized.shape[1]) // 2
    background[y_offset:y_offset + image_resized.shape[0], x_offset:x_offset + image_resized.shape[1]] = image_resized
    return background


def save_concat_image_with_labels(images, labels, output_path, images_per_row=7):
    outer_margin = 50
    gap = 10
    label_height = 40

    image_height, image_width = images[0].shape[:2]
    total_rows = len(images) // images_per_row
    total_width = outer_margin + images_per_row * image_width + (images_per_row - 1) * gap + outer_margin
    total_height = outer_margin + total_rows * (image_height + label_height + gap) - gap + outer_margin

    concat_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # 先把小图拼到 concat_image
    for idx, image in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x_offset = outer_margin + col * (image_width + gap)
        y_offset = outer_margin + row * (image_height + label_height + gap)
        concat_image[y_offset:y_offset + image.shape[0],
                     x_offset:x_offset + image.shape[1]] = image

    # === 将 NumPy -> PIL 来画文字 ===
    pil_img = Image.fromarray(concat_image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    # 在每列最下方加标签
    for col in range(images_per_row):
        if col < len(labels):
            label = labels[col]
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]

            text_x = outer_margin + col * (image_width + gap) + (image_width - text_width) // 2
            text_y = total_height - outer_margin - label_height

            # 这里要在循环体内调用draw.text，让每个列都能绘制文字
            # draw.text((text_x, text_y), label, font=font, fill=(0,0,0))

    # === 最终保存带文字的 PIL 图像 ===
    pil_img.save(output_path)
    print(f"Grad-CAM heatmap with labels saved at: {output_path}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输入图片路径
    image_paths = [

    ]

    # 模型参数路径
    model_paths = {

    }

    # model对象
    models_dict = {

    }

    # 可视化选择的层
    target_layers = {

    }

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_with_background = create_white_background_image(original_image, 360, 360)
        all_images.append(original_image_with_background)

        for model_name in list(model_paths.keys())[1:]:
            model = models_dict[model_name]
            model.load_state_dict(torch.load(model_paths[model_name], map_location=device))
            model.to(device)
            model.eval()
            target_layer = target_layers[model_name](model)
            cam = generate_gradcam(model, image_tensor, target_layer, device)
            heatmap = overlay_heatmap(image_path, cam)
            heatmap_with_background = create_white_background_image(heatmap, 360, 360)
            all_images.append(heatmap_with_background)

            # === 在这里单独保存每个模型的热力图，文件名使用模型名 ===
            save_dir = r''
            model_heatmap_path = os.path.join(save_dir, f"{model_name}.png")
            cv2.imwrite(model_heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            # print(f"单独保存 {model_name} 的热力图到: {model_heatmap_path}")

    output_path = os.path.join(os.getcwd(), 'test', "gradcam_all_images.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_concat_image_with_labels(all_images, list(model_paths.keys()), output_path, images_per_row=7)


if __name__ == "__main__":
    main()



