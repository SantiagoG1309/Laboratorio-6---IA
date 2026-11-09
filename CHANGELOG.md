# Historial de cambios

## v1.1.1 — 2026-11-08

### Aplicación
- Pestaña **Arquitectura** adaptable: sin desplazamiento horizontal a 1024 × 620, controles que
  se reacomodan, tabla con columnas elásticas y diagrama que se ajusta al ancho.
- **Laboratorio numérico** rehecho con los pesos aprendidos del modelo cargado y una muestra real
  del dataset preprocesada: convolución de un filtro, producto punto de una neurona oculta y
  Softmax de salida, paso a paso.
- Las antiguas capas «densas» ahora se llaman **capas ocultas** en el núcleo, la tabla, el
  diagrama, el inspector, el glosario y la documentación.
- **Pipeline**: se eliminaron los interruptores duplicados de la derecha; cada tarjeta de etapa
  conserva su propio control.
- **Entrenamiento**: tooltips explicativos en división de datos, modelo, arquitectura, métricas,
  gráfica, progreso y registro; botón **Aplicar configuración recomendada**.

### Distribución
- Versión de Windows `1.1.1.0` y capturas actualizadas.

## v1.1.0 — 2026-11-08

### Aplicación
- Ventana mínima de 1024 × 620 y barra lateral adaptable; tarjetas y gráficos revisados para
  pantallas pequeñas y HiDPI.
- La copia inicial del dataset de ejemplo y las miniaturas de predicciones se ejecutan en
  segundo plano, sin congelar la interfaz.
- El cierre de la ventana detiene los hilos de entrenamiento de forma ordenada.
- Tablas y diagramas usan los colores del tema activo (incluido el modo claro/oscuro).
- Corrección del importación de imágenes en la vista de arquitectura y del estado del modelo
  incluido.

### Núcleo
- Validación y normalización de las configuraciones guardadas en JSON (tipos, rangos y kernels).
- Lectura/escritura de imágenes robusta para rutas no ASCII y arreglos `float` en `[0, 1]`.
- División estratificada garantiza conjuntos train/val/test no vacíos.
- Métricas vectorizadas y una sola pasada de métricas por clase.
- Comprobaciones explícitas de formas, canales y kernels en la CNN; `gradient_check` admite
  ambas APIs.
- Formato de modelo v2 con nombres y formas de tensores para detectar modelos incompatibles.
- Caché del pipeline protegida con bloqueo, tiempos de lectura reales y etapa final coherente
  con el tensor de entrenamiento.
- El entrenamiento omite imágenes corruptas, predice por lotes y limita la evaluación de train
  en datasets grandes.
- Exportación con nombres únicos, copia del original sin recomprimir y CSV UTF-8 con BOM.

### Distribución
- Ejecutable portable con dataset, modelo preentrenado, `COMO_USAR.txt`, `LEEME.md`, README y
  capturas.
- Dependencia `openpyxl` incluida para exportar la arquitectura a Excel.
- ZIP con los archivos en la raíz (sin carpeta envolvente) y versión de Windows `1.1.0.0`.
- Documentación de uso, solución de problemas y créditos actualizada.

## v1.0.0 — 2026-11-08
- Primera versión del laboratorio: pipeline visual, arquitectura CNN, entrenamiento con
  Backpropagation + Momentum, resultados y exportación.
