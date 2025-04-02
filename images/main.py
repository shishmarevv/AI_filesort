import infer

if __name__ == "__main__":
    path = "image_path"
    predict = infer.classify_image(path)
    print(f"Predicted image class: {predict}")