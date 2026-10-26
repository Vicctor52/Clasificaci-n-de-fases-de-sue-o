# Clasificación de fases del sueño

Este proyecto entrena un modelo para clasificar automáticamente las fases del sueño a partir de registros de polisomnografía (PSG). La señal se divide en fragmentos de 30 segundos, llamados épocas, y se extraen características de cada época para entrenar el clasificador.

Las anotaciones de dos expertos se utilizan como referencia para entrenar y evaluar el sistema. El objetivo es predecir cinco clases: Wake, N1, N2, N3 y REM.

Este proyecto es un ejercicio académico de procesamiento de señales y aprendizaje automático. No es un dispositivo médico ni ha sido validado para uso clínico, por lo que sus resultados no deben utilizarse para diagnosticar ni tomar decisiones médicas.

## Origen de los datos

Los registros proceden de **ISRUC-Sleep**, un conjunto público de polisomnografías de noche completa con señales fisiológicas y anotaciones de fases del sueño realizadas por dos expertos. Para este proyecto se utiliza la **Cohorte III**, formada por diez personas sanas.

Los datos originales no se incluyen en este repositorio: descárgalos desde la [página oficial de ISRUC-Sleep](https://sleeptight.isr.uc.pt/index.html@p=48.html) y consulta sus condiciones de uso antes de utilizarlos o redistribuirlos.

Cada participante debe colocarse en una subcarpeta numerada dentro de `data/`, con el registro `.edf` y las anotaciones de los dos expertos:

```text
data/
├── 1/
│   ├── 1.edf
│   ├── 1_1.txt
│   └── 1_2.txt
├── 2/
│   ├── 2.edf
│   ├── 2_1.txt
│   └── 2_2.txt
└── ... hasta 10/
```

No subas los registros ni las anotaciones originales al repositorio. La disponibilidad pública para investigación no implica necesariamente permiso para volver a publicar los archivos.

## Estructura del proyecto

```text
fases_de_sueño/
├── README.md
├── requirements.txt
├── data/                  # Crear localmente; dataset no incluido
└── src/
    └── config.py          # Canales, clases y parámetros del flujo
```

## Dependencias

Las dependencias se instalan desde `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```
