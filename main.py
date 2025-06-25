from src.data_loader import load_mnist
import matplotlib.pyplot as plt

def show_samples(images, labels, num=9):
    """
    Показывает первые N изображений с метками.
    """
    plt.figure(figsize=(6,6))
    for i in range(num):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train, X_test, y_test = load_mnist(normalize=True, flatten=False)

    print(f"Train images shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test images shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    show_samples(X_train, y_train)

if __name__ == "__main__":
    main()
