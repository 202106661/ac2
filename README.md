# Urban Sound Classification with Deep Learning

Este repositório contém a implementação de modelos de aprendizado profundo para classificação de sons urbanos utilizando o dataset [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html). O projeto foi desenvolvido como parte da disciplina **Machine Learning II (CC3043)**, DCC/FCUP (2024/2025).

---

## Objetivo
O objetivo deste projeto é construir classificadores capazes de identificar sons urbanos em uma das seguintes 10 classes:
- Ar condicionado
- Buzina de carro
- Crianças brincando
- Latido de cachorro
- Furadeira
- Motor em marcha lenta
- Disparo de arma
- Martelo hidráulico
- Sirene
- Música de rua

---

## Dataset
O [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) contém 8.732 amostras de áudio rotuladas, com duração de até 4 segundos. Os áudios estão organizados em 10 classes diferentes. As descrições detalhadas do dataset podem ser encontradas [neste artigo](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf).

---

## Implementação
O projeto implementa dois classificadores baseados em aprendizado profundo:
1. **Multilayer Perceptron (MLP)**
2. **Convolutional Neural Network (CNN)**

### Etapas de Implementação
1. **Pré-processamento e Preparação dos Dados**:
   - Normalização e uniformização dos áudios.
   - Extração de características (e.g., MFCCs, energia RMS, entre outros) usando a biblioteca [Librosa](https://librosa.org/).

2. **Definição da Arquitetura dos Modelos**:
   - **MLP**:
     - Número de camadas e neurônios.
     - Funções de ativação.
   - **CNN**:
     - Arquitetura baseada em representações 2D de áudio (e.g., espectrogramas de Mel).

3. **Treinamento dos Modelos**:
   - Estratégias de treinamento:
     - Otimizador: `Adam`.
     - Regularização: `Dropout` e `Early Stopping`.
     - Hiperparâmetros: Taxa de aprendizado, tamanho de batch, número de épocas.

4. **Avaliação de Desempenho**:
   - Validação cruzada com 10 folds.
   - Uso de matriz de confusão e cálculo da acurácia média.

---

## Resultados
Os modelos foram avaliados usando validação cruzada com 10 folds. A performance foi quantificada com base na acurácia média e desvio padrão. 

- **MLP**: Acurácia média de XX.XX%.
- **CNN**: Acurácia média de XX.XX%.

---

## Bônus
Foi implementada a abordagem **DeepFool** para avaliar a robustez dos modelos contra exemplos adversariais. Essa técnica encontra a menor perturbação necessária para enganar o classificador.

---

## Como Executar
1. **Pré-requisitos**:
   - Python 3.7+
   - Bibliotecas: `numpy`, `pandas`, `librosa`, `tensorflow`, `sklearn`

2. **Clone este repositório**:
   ```bash
   git clone https://github.com/seu_usuario/urban-sound-classification.git
   cd urban-sound-classification

3. **Baixe o dataset UrbanSound8K**:
   https://urbansounddataset.weebly.com/urbansound8k.html

4. **Coloque os arquivos na pasta data/UrbanSound8K**

5. **Execute o script de pré-processamento**:
   ```bash
   python features.py
   
6. **Treine os modelos**:
Utilize o notebook UrbanSounds.ipynb para treinar e avaliar os modelos.
   
---

## Contribuidores
- Ana Catarina Ribeiro
- Estefany Vasconcelos
- Margarida Sousa
