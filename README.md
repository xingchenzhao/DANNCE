# Domain Adversarial Neural Networks for Domain Generalization

This repository hosts the implementation for our paper, ["Domain Adversarial Neural Networks for Domain Generalization: When It Works and How to Improve"](https://arxiv.org/pdf/2102.03924.pdf). Our approach explores the application of domain adversarial neural networks for domain generalization.

## Getting Started

### Prerequisites

To run the code, you first need to set up the environment. We use Conda for managing dependencies. The environment can be set up using the provided `environment.yml` file with the following command:

```bash
conda env create --file environment.yml
```

### Data and Pretrained Model

You will also need to download the pretrained AlexNet model `pretrained_alexnet.pth`. You can find this file in the `Files` section at the following URL:

[https://osf.io/87tjs/?view_only=99ca1354ca844d26be26516281ce964d](https://osf.io/87tjs/?view_only=99ca1354ca844d26be26516281ce964d)

Once you have downloaded the file, place it in the `src/models/caffenet` directory.

### Running the Experiments

To execute the experiments, navigate to the directory containing `final_scripts` and `src`. You can then run the bash scripts located in `final_scripts/pacs` and `final_scripts/office_home`.

We have included scripts for all of our main experiments in the paper. If you wish to run additional experiments, you can check the arguments that `src.main` accepts. This can be done either by looking at the code or using the help argument as follows:

```bash
python3 -m src.main --help
```

## Contributing

We welcome any contributions to improve this project. Feel free to open issues or make pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Citation

If you find this work useful in your research, please consider citing our paper.

## Contact

For any questions, please open an issue and we'll get back to you as soon as possible.