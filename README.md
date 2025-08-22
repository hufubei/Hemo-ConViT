# Hemo-ConViT: Anemia Detection using Computer Vision

This project uses a Vision Transformer (ViT) with adaptive contrastive learning to predict hemoglobin (Hgb) levels from images of the eye's conjunctiva.

## Project Structure
HemaConViT/

├── .gitignore

├── README.md

├── requirements.txt

├── config.py

└── src/

├── init.py

├── data_loader.py

├── model.py

├── utils.py

├── train.py

└── evaluate.py


## Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/HemaConViT.git
    cd HemoConViT
    ```

2.  Create a virtual environment and install the dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Data Preparation**:
    - Download the image dataset and place it in a `data/images/` folder.
    - Place the `total_defy.xlsx` metadata file in the `data/` folder.

## Usage

### Training

To train the model, run the `train.py` script. Make sure the paths in `config.py` point to your data.

```bash
python src/train.py
 ```

Model checkpoints and metric plots will be saved to the directories specified in config.py.

### Evaluation

To evaluate a pre-trained model checkpoint:

```bash
python src/evaluate.py --model_path /path/to/your/model.h5 --threshold 12.0
```


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

jose humberto fuentes-beingolea
universidad nacional de san antonio abad del cusco
