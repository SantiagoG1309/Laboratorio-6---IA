# AquaCNN · Laboratorio 6 — Guía rápida

**Clasificación de peces (trucha, mojarra y pavón) con una red neuronal convolucional.**

## Cómo abrir la aplicación

1. Descomprime **todo** el ZIP en una carpeta normal (Escritorio, Documentos, etc.).
   No ejecutes `AquaCNN.exe` desde dentro del ZIP.
2. Haz doble clic en **`AquaCNN.exe`**.
3. Espera unos segundos la primera vez: la app prepara el dataset de ejemplo y carga
   el modelo ya entrenado. El resto de arranques son inmediatos.

No hace falta instalar Python, ni conexión a internet, ni permisos de administrador.

## Aviso de Windows SmartScreen

La aplicación no está firmada digitalmente. Si aparece el aviso:

**Más información → Ejecutar de todas formas**

## Dónde se guardan tus datos

La app intenta guardar el dataset, los modelos y las exportaciones junto al `.exe`.
Si esa carpeta no permite escritura (por ejemplo, `C:\Program Files`), usa
`%LOCALAPPDATA%\AquaCNN` automáticamente.

## Primeros pasos

- **Inicio**: muestra las 341 imágenes incluidas, su distribución y la división
  entrenamiento / validación / prueba (70 / 15 / 15).
- **Pipeline visual**: observa paso a paso cómo se procesa una imagen (color, segmentación
  Otsu, escala de grises, mejora de contraste —histograma o CLAHE—, convolución, pooling y
  entrada a la red).
- **Arquitectura NN**: diagrama de la red, tablas de formas y parámetros, y un
  laboratorio numérico que desglosa la convolución, una neurona oculta y el Softmax
  con los pesos reales del modelo y una imagen del dataset. Pasa el cursor por cada
  recuadro del diagrama para ver su operación, forma de salida y fórmula.
- **Entrenamiento**: ajusta épocas, batch, learning rate, filtros, método de contraste, etc.
  y pulsa **Iniciar entrenamiento**. El proceso es secuencial: entrena, valida las
  instantáneas guardadas y al final evalúa la prueba. El progreso se ve en vivo. El botón
  **Aplicar configuración recomendada** carga el preset que mejor equilibró las tres clases.
- **Resultados**: exactitud, matriz de confusión, métricas por clase, predicciones
  y exportación de reportes.
- **Predicción**: arrastra cualquier foto de un pez y obtén la clase estimada con su
  porcentaje de confianza y las probabilidades de las tres clases.
- **Configuración**: temas, color de acento, tipografía y comportamiento.

El modelo incluido ya está evaluado al abrir la app, así que **Resultados** muestra
métricas reales sin necesidad de entrenar.

## Añadir tus propias imágenes

En **Inicio**, usa **Añadir imágenes** en cada tarjeta de clase, o arrastra archivos
a la zona de carga. También puedes elegir otra carpeta con **Cargar dataset**.

Estructura esperada:

```
Dataset Imagenes/
├── trucha/
├── mojarra/
└── pavon/
```

Los nombres deben ser exactamente `trucha`, `mojarra` y `pavon` (sin tildes).

## Atajos

| Atajo | Acción |
|---|---|
| `Ctrl + 1…8` | Cambiar de página |
| `Ctrl + O` | Elegir carpeta del dataset |
| `F5` | Recalcular el pipeline |
| `Ctrl + E` | Exportar el dataset procesado |
| `F11` | Pantalla completa |
| `Esc` | Salir de pantalla completa |

## Solución de problemas

- **No abre / se cierra al instante**: comprueba que descomprimiste la carpeta
  completa y que `AquaCNN.exe` está junto a la carpeta `_internal`.
- **SmartScreen bloquea el archivo**: usa *Más información → Ejecutar de todas formas*.
- **Antivirus lo pone en cuarentena**: es un falso positivo habitual de PyInstaller;
  añade una exclusión para la carpeta de AquaCNN.
- **Quiero empezar de cero**: borra `%LOCALAPPDATA%\AquaCNN` y vuelve a abrir la app.

## Créditos

Universidad de Cundinamarca · Facultad de Ingeniería · Sede Facatativá.

Autores: Kevin Santiago García Ospina, Julián Andrés Pérez Mendigaño,
Iván Daniel Palacio Hernández y Andrés Felipe Maldonado Forero.

Dataset: Wikimedia Commons e iNaturalist (licencias Creative Commons).
Consulta `Dataset Imagenes/CREDITOS.txt` para la atribución completa.
