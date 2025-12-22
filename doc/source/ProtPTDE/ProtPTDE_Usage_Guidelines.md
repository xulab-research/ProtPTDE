# ProtPTDE: Protein Pre-Training Model-Assisted Directed Evolution



**ProtPTDE** is a computational framework that accelerates protein directed evolution by integrating multiple deep learning models. Our solution features **unified parameter governance** through a single configuration file and **extensible architecture** supporting custom protein language models and multi-model embeddings.

##  Key Features
- **Centralized Configuration**: All parameters managed in `config/config.json`
- **Multi-Model Integration**: Concatenate embeddings from multiple protein language models
- **Extensible Design**: Easily add custom protein language models
- **Automated Workflow**: From data processing to fitness prediction and cluster analysis

## Installation

1. **Create conda environment**
```bash
conda create -n Prot_PTDE python=3.13 -y
conda activate Prot_PTDE
```

2. **Install dependencies**
```bash
pip install torch==2.7.1 tqdm click biopython "pandas[excel]" scikit-learn more-itertools iterative-stratification optuna transformers einops seaborn plotly
conda install numba -y
```

##  Required Input Files
> **Note**: Prepare these files before starting the workflow
1. **Mutation Data**: `.xlsx` file in `01_data_processing/` directory containing mutations and fitness values
2. **Wild-type Sequence**: FASTA file named `result.fasta` in `features/wt/` directory

>  **Important**: Before execution, rename bash scripts in `02_cross_validation/` and `03_final_model/` folders to match your GPU configuration. The last digit indicates GPU card number (e.g., `2000.sh` → GPU 0).

##  Workflow

### 1. Data Processing & Embedding Generation
*Converts mutation data to standardized format and generates protein embeddings from multiple pre-trained models.*

1. **Configure parameters** in `config.json` :  

   Set `"basic_data_name"` to your data file name (without extension)

2. **Execute commands**
```bash
cd 01_data_processing
python 01_convert_and_generate_fasta_file.py
cd ..

cd generate_features
python generate_all_embeddings.py
cd ..
```

3. **Proceed to next steps** : 
   Continue with cross validation after embeddings are generated

### 2. Cross Validation
*Identifies optimal model architectures and hyperparameters through systematic validation.*

1. **Configure parameters** in `config.json` :

   Set `"single_model_embedding_output_dim"` ;
   <br />
   Adjust `"cross_validation"` parameters ;
   <br />
   Define hyperparameter search ranges under `"cross_validation.hyperparameter_search"` .

2. **Execute commands**
```bash
cd 02_cross_validation
bash 01_train.sh
python 02_Dis_cross_validation.py  # Generates Dis_cross_validation.pdf
cd ..
```

3. **Update hyperparameters**
   <br />
   Based on Dis_cross_validation.pdf, update the `selected_models`, `num_layer`, and `max_lr` parameters in the `best_hyperparameters` section

### 3. Training & Fine-tuning
*Builds robust predictive models through ensemble training and targeted fine-tuning.*

1. **Configure parameters** in `config.json`  

   Adjust training parameters under `"final_model.train_parameter"`;
   <br />Set fine-tuning parameters under `"final_model.finetune_parameter"`.

2. **Execute commands**
```bash
cd 03_final_model
bash 01_train.sh
python 02_plot_random_seed_train.py  # Generates Scatter_best_train_test_epoch_ratio.html
bash 03_finetune.sh
cd ..
```

3. **Select optimal seed**
   <br />Based on Scatter_best_train_test_epoch_ratio.html, update the `random_seed` parameter in the `best_hyperparameters` section

### 4. Inference & Clustering
*Predicts fitness for novel mutations and clusters results by prediction reliability.*

1. **Configure parameters** in `config.json` : 

   Set `"max_mutations"` under `"inference"`

2. **Execute commands**
```bash
cd 04_inference
bash 01_generate_unpredicted_muts_csv.sh
bash 02_inference.sh  # Outputs reliability-ranked predictions
cd ..
```

3. **Analyze results**
   <br />Review the predictions ranked by reliability in the output directory

## Adding Custom Models (if needed)
*To integrate your own protein language model:*

1. **Create model directory structure**:
   ```bash
   mkdir -p generate_features/your_model_embedding
   touch generate_features/your_model_embedding/function.py
   ```

2. **Implement model functions** in `generate_features/your_model_embedding/function.py`:

   Ensure your `function.py` includes **exactly two functions** with the following specifications:

   - **Model loader function**:  `get_[yourmodelname]_model() → tuple`  
     *Loads and caches your model instance using singleton pattern*

   - **Embedding generator function**:  `generate_[yourmodelname]_embedding(sequence: str) → torch.Tensor`  
     *Generates per-residue embeddings; must return tensor of shape `[sequence_length, embedding_dimension]`*

 

   > Function names must follow this pattern with your model name replacing `[yourmodelname]` (e.g., `get_esm2_model` and `generate_esm2_embedding`)
  
