# ant-simu-com-NEAT
# 🐜 Simulação de Formigueiro com Evolução NEAT

Este é um projeto de simulação de um formigueiro onde **formigas evoluem com redes neurais e algoritmos genéticos (NEAT)** para encontrar comida e levá-la de volta ao ninho. A simulação é visualizada com `pygame` e evolui ao longo de várias gerações.

## 🎯 Objetivo

Desenvolver uma inteligência artificial capaz de controlar formigas de forma eficiente para:
- Explorar o ambiente
- Encontrar comida
- Levar comida ao ninho
- Deixar rastros com feromônio
- Evoluir geração após geração com base no desempenho

---

## 🧠 Tecnologias e Bibliotecas

- Python 3.12+
- [pygame](https://www.pygame.org/) – Visualização gráfica
- [neat-python](https://github.com/CodeReclaimers/neat-python) – Algoritmo genético NEAT
- Algoritmos personalizados de simulação e lógica de evolução

---

## 🧪 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seuusuario/formigueiro-evolutivo.git
cd formigueiro-evolutivo

2. Crie um ambiente virtual (opcional, mas recomendado)

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows3. Instale as dependências

pip install -r requirements.txt

Ou manualmente:

pip install pygame neat-python

4. Execute a simulação

python formigueiro.py

🕹️ Controles e Visualização

    Formigas são exibidas como pontos pretos.

    O ninho é o ponto vermelho no centro.

    A comida aparece como pontos verdes no mapa.

    Feromônios deixados pelas formigas são visíveis em azul claro.

A cada geração, as formigas aprendem com tentativas anteriores.
⚙️ Configurações

O arquivo neat-config.txt define os parâmetros de evolução:

    Número de inputs/outputs da rede neural

    Taxa de mutação

    Tamanho da população

    Estagnação máxima

    Número de gerações

📈 Critérios de Evolução

    Fitness aumenta ao levar comida ao ninho.

    Penalização leve por distância da comida.

    Formigas que encontram e retornam com comida têm maior chance de reprodução (clonagem).

    Pheromônios ajudam outras formigas a seguir trilhas de sucesso.

📌 Melhorias Futuras

    Visão local com sensores (matriz 3x3 ao redor da formiga)

    Obstáculos no mapa

    Níveis com dificuldade crescente

    Limite de carregamento de comida

    Salvar e carregar o melhor modelo

📸 Exemplo de Simulação

🧑‍💻 Autor

Projeto desenvolvido por Ademir Neto
