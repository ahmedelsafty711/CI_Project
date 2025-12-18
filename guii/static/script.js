
document.addEventListener('DOMContentLoaded', () => {
    const svg = document.getElementById('nn-svg');
    const predictionOutput = document.getElementById('prediction-output');
    const inputButtons = document.querySelectorAll('.input-btn');
    const predictBtn = document.getElementById('predict-btn');

    let currentInput = [-1, -1];

    // --- UTILITY FUNCTIONS ---
    const activationToColor = (value) => {
        // Maps value from [-1, 1] to a blue-to-red color scale
        const normalized = (value + 1) / 2; // to [0, 1]
        const hue = (1 - normalized) * 240; // 240 (blue) to 0 (red)
        return `hsl(${hue}, 80%, 50%)`;
    };
    
    // --- EVENT LISTENERS ---
    inputButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.dataset.inputIndex, 10);
            const currentValue = parseInt(btn.dataset.value, 10);
            const newValue = currentValue === 1 ? -1 : 1;

            btn.dataset.value = newValue;
            btn.textContent = `Input ${index + 1}: [${newValue}]`;
            btn.classList.toggle('active', newValue === 1);
            
            currentInput[index] = newValue;
        });
    });

    predictBtn.addEventListener('click', () => {
        getPrediction(currentInput);
    });

    // --- API CALLS & RENDERING ---
    const drawNetwork = (layers) => {
        svg.innerHTML = '';
        const neuronRadius = 15;
        const layerGap = 120;
        const neuronGap = 40;

        let maxNeurons = 0;
        layers.forEach(layer => {
            if (layer.type === "Dense" || layer.type === "Tanh" || layer.type === "Sigmoid") {
                const numNeurons = layer.output_size || layer.input_size || 1;
                if (numNeurons > maxNeurons) maxNeurons = numNeurons;
            }
        });
        
        const svgHeight = maxNeurons * neuronGap + 20;
        const svgWidth = (layers.length) * layerGap;
        svg.setAttribute('height', svgHeight);
        svg.setAttribute('width', svgWidth);

        let prevLayerCoords = [];
        let layer_idx = -1;

        // Draw input layer representation
        const inputLayer = layers[0];
        const inputSize = inputLayer.input_size;
        let currentLayerCoords = [];
        const inputLayerHeight = (inputSize - 1) * neuronGap;
        for (let i = 0; i < inputSize; i++) {
            const y = (svgHeight - inputLayerHeight) / 2 + (i * neuronGap);
            const x = layerGap / 2;
            currentLayerCoords.push({ x, y });
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', neuronRadius);
            circle.setAttribute('fill', '#f0f2f5');
            circle.setAttribute('stroke', '#ccc');
            circle.setAttribute('class', `neuron input-neuron-${i}`);
            svg.appendChild(circle);
        }
        prevLayerCoords = currentLayerCoords;

        layers.forEach(layer => {
            if (layer.type !== "Dense") return; // Only draw Dense layers explicitly
            
            layer_idx++;
            currentLayerCoords = [];
            const numNeurons = layer.output_size;
            const layerHeight = (numNeurons - 1) * neuronGap;
            
            // Draw neurons
            for (let i = 0; i < numNeurons; i++) {
                const y = (svgHeight - layerHeight) / 2 + (i * neuronGap);
                const x = layerGap / 2 + (layer_idx + 1) * layerGap;
                currentLayerCoords.push({ x, y });
                
                // Draw connections from previous layer
                prevLayerCoords.forEach(prevCoord => {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', prevCoord.x);
                    line.setAttribute('y1', prevCoord.y);
                    line.setAttribute('x2', x);
                    line.setAttribute('y2', y);
                    line.setAttribute('stroke', '#ddd');
                    line.setAttribute('class', `connection l${layer_idx}-${i}`);
                    svg.appendChild(line);
                });
            }
             // Draw neurons on top of lines
             for (let i = 0; i < numNeurons; i++) {
                const {x, y} = currentLayerCoords[i];
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', x);
                circle.setAttribute('cy', y);
                circle.setAttribute('r', neuronRadius);
                circle.setAttribute('fill', '#f0f2f5');
                circle.setAttribute('stroke', '#ccc');
                circle.setAttribute('class', `neuron layer-${layer_idx}-neuron-${i}`);
                svg.appendChild(circle);
            }
            
            prevLayerCoords = currentLayerCoords;
        });
    };

    const updateVisualization = (input, activations) => {
        // Update input neurons
        input.forEach((val, i) => {
            const circle = svg.querySelector(`.input-neuron-${i}`);
            if (circle) circle.setAttribute('fill', activationToColor(val));
        });

        // Update hidden and output neurons
        let activationLayerIdx = 0;
        let denseLayerIdx = 0;
        
        window.networkSchema.layers.forEach(layer => {
            if(layer.type === "Dense") {
                 const layerActivations = activations[activationLayerIdx];
                 layerActivations.forEach((val, i) => {
                    const circle = svg.querySelector(`.layer-${denseLayerIdx}-neuron-${i}`);
                    if (circle) circle.setAttribute('fill', activationToColor(val));
                });
                denseLayerIdx++;
            }
            if(layer.type === "Tanh" || layer.type === "Sigmoid"){
                activationLayerIdx++;
            } else if (layer.type === "Dense"){
                 activationLayerIdx++;
            }
        });
    };

    const getPrediction = async (input) => {
        try {
            const response = await fetch('/api/xor/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: input })
            });
            if (!response.ok) throw new Error('Network request failed');

            const data = await response.json();
            predictionOutput.textContent = data.prediction[0].toFixed(4);
            updateVisualization(input, data.activations);

        } catch (error) {
            console.error('Prediction error:', error);
            predictionOutput.textContent = 'Error';
        }
    };

    const initialize = async () => {
        try {
            const response = await fetch('/api/xor/network');
            if (!response.ok) throw new Error('Failed to fetch network structure');
            
            const data = await response.json();
            window.networkSchema = data;
            drawNetwork(data.layers);
        } catch (error) {
            console.error('Initialization error:', error);
        }
    };

    initialize();
});
