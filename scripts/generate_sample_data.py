"""
Sample data generation script for Product Fraud Detection System
This creates realistic sample product images for testing
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import config


def create_product_image(product_id, color, shape='rectangle', size=config.IMAGE_SIZE):
    """
    Create a synthetic product image
    
    Args:
        product_id: Unique identifier for the product
        color: RGB tuple for product color
        shape: Shape of the product ('rectangle', 'circle', 'triangle')
        size: Image size tuple
        
    Returns:
        PIL Image object
    """
    # Create white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Calculate dimensions
    margin = 40
    width, height = size
    
    # Draw product shape
    if shape == 'rectangle':
        draw.rectangle(
            [margin, margin, width-margin, height-margin],
            fill=color,
            outline=(0, 0, 0),
            width=3
        )
    elif shape == 'circle':
        draw.ellipse(
            [margin, margin, width-margin, height-margin],
            fill=color,
            outline=(0, 0, 0),
            width=3
        )
    elif shape == 'triangle':
        points = [
            (width//2, margin),
            (margin, height-margin),
            (width-margin, height-margin)
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    
    # Add product ID text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = f"ID: {product_id}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_position = ((width - text_width) // 2, (height - text_height) // 2)
    
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    
    return img


def add_damage_to_image(img, damage_level='minor'):
    """
    Add simulated damage to a product image
    
    Args:
        img: PIL Image object
        damage_level: 'minor', 'major', or 'severe'
        
    Returns:
        Damaged PIL Image
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    if damage_level == 'minor':
        # Add small scratches
        for _ in range(5):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = x1 + np.random.randint(-30, 30)
            y2 = y1 + np.random.randint(-30, 30)
            draw.line([x1, y1, x2, y2], fill=(50, 50, 50), width=2)
    
    elif damage_level == 'major':
        # Add dents and larger scratches
        for _ in range(10):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = x1 + np.random.randint(-50, 50)
            y2 = y1 + np.random.randint(-50, 50)
            draw.line([x1, y1, x2, y2], fill=(30, 30, 30), width=4)
        
        # Add dark spots
        for _ in range(5):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(10, 30)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(40, 40, 40))
    
    elif damage_level == 'severe':
        # Add cracks
        for _ in range(15):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = x1 + np.random.randint(-80, 80)
            y2 = y1 + np.random.randint(-80, 80)
            draw.line([x1, y1, x2, y2], fill=(0, 0, 0), width=5)
        
        # Add large damaged areas
        for _ in range(8):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(20, 50)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(20, 20, 20))
        
        # Apply blur to simulate severe damage
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


def generate_sample_dataset():
    """
    Generate a complete sample dataset for testing
    """
    print("Generating sample product dataset...")
    
    # Define product catalog
    products = [
        {'id': 'PROD001', 'color': (255, 100, 100), 'shape': 'rectangle', 'name': 'Red Box'},
        {'id': 'PROD002', 'color': (100, 100, 255), 'shape': 'circle', 'name': 'Blue Ball'},
        {'id': 'PROD003', 'color': (100, 255, 100), 'shape': 'triangle', 'name': 'Green Triangle'},
        {'id': 'PROD004', 'color': (255, 255, 100), 'shape': 'rectangle', 'name': 'Yellow Box'},
        {'id': 'PROD005', 'color': (255, 100, 255), 'shape': 'circle', 'name': 'Purple Ball'},
    ]
    
    # Create directories
    original_dir = config.ORIGINAL_PRODUCTS_DIR
    sample_dir = config.SAMPLE_DATA_DIR
    
    os.makedirs(os.path.join(sample_dir, 'authentic_returns'), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'fraudulent_returns'), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'damaged_products'), exist_ok=True)
    
    # Generate original product images
    print("\nGenerating original product images...")
    for product in products:
        img = create_product_image(product['id'], product['color'], product['shape'])
        save_path = os.path.join(original_dir, f"{product['id']}.png")
        img.save(save_path)
        print(f"  Created: {product['name']} -> {save_path}")
    
    # Generate authentic returns (same product, slight variations)
    print("\nGenerating authentic return samples...")
    for product in products:
        img = create_product_image(product['id'], product['color'], product['shape'])
        # Add slight noise to simulate different lighting/angle
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        save_path = os.path.join(sample_dir, 'authentic_returns', f"{product['id']}_return.png")
        img.save(save_path)
        print(f"  Created: {product['name']} return -> {save_path}")
    
    # Generate fraudulent returns (different products)
    print("\nGenerating fraudulent return samples...")
    for i, product in enumerate(products):
        # Use a different product's characteristics
        fraud_product = products[(i + 2) % len(products)]
        img = create_product_image(product['id'], fraud_product['color'], fraud_product['shape'])
        
        save_path = os.path.join(sample_dir, 'fraudulent_returns', f"{product['id']}_fraud.png")
        img.save(save_path)
        print(f"  Created: Fraudulent return for {product['name']} -> {save_path}")
    
    # Generate damaged products
    print("\nGenerating damaged product samples...")
    damage_levels = ['minor', 'major', 'severe']
    
    for product in products:
        for damage_level in damage_levels:
            img = create_product_image(product['id'], product['color'], product['shape'])
            damaged_img = add_damage_to_image(img, damage_level)
            
            save_path = os.path.join(
                sample_dir, 
                'damaged_products', 
                f"{product['id']}_{damage_level}.png"
            )
            damaged_img.save(save_path)
            print(f"  Created: {product['name']} ({damage_level} damage) -> {save_path}")
    
    print("\n" + "="*80)
    print("Sample dataset generation complete!")
    print("="*80)
    print(f"\nOriginal products: {original_dir}")
    print(f"Sample data: {sample_dir}")
    print(f"\nTotal images created: {len(products) * (1 + 1 + 1 + len(damage_levels))}")


if __name__ == '__main__':
    generate_sample_dataset()
