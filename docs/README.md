# YOLO_Bicycle

## Use YOLO_Bicycle
For use YOLO_Bicycle, clone the repository and run:

```
make install
make run
```

## Transfer Learning

O aprendizado por transferência(ou transfer learning, em inglês) é uma técnica fundamental em aprendizado de máquina, especialmente quando lidamos com conjuntos de dados limitados. Essa técnica se baseia na ideia de reutilizar o conhecimento adquirido por modelos pré-treinados em tarefas específicas para resolver problemas relacionados. Por exemplo, se uma rede neural artificial previamente treinada foi bem-sucedida na resolução de um problema, seus conhecimentos podem ser transferidos e adaptados para resolver um problema similar.

Caso seja necessário aplicar a técnica de aprendizado por transferência em uma rede neural profunda pré-treinada, basta remover sua camada totalmente conectada, transformando-a em um extrator de características. Essas características extraídas podem ser utilizadas como entrada para uma nova rede neural, a qual é re-treinada com um novo conjunto de dados para resolver um problema diferente ou semelhante. Esse processo permite uma adaptação eficiente do modelo pré-existente para novos desafios, economizando tempo e recursos computacionais.

No código desenvolvido foi aplicado o modelo de transfer learning, foi utilizado um modelo pré-treinado do próprio YOLO, isso resultou em um desenvolvimento mais eficiente, visto que não foi necessário treinar o modelo, e apresentou-se preciso, pois ao utilizar vídeo nos primeiros frames onde a roda da bicicleta surgia o modelo já identificava o objeto.

REFERÊNCIA:

[1] ACAR, E.; YILMAZ, İ. COVID-19 detection on IBM quantum computer with classical-quantum transfer learning. TURKISH JOURNAL OF ELECTRICAL ENGINEERING & COMPUTER SCIENCES, v. 29, n. 1, p. 46–61, 27 jan. 2021.
