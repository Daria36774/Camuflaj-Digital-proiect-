# Camuflaj Digital în Imagini Termice: Evaluarea Detectabilității Persoanelor

**Autori:** Grigoraș Ioana-Daria & Moscalu Ionela-Georgiana  
**Grupa:** 1307B  
**Materia:** Prelucrarea Imaginilor (PI) — Proiect de Laborator (PI_P_lab2)  

---

## Descrierea Scurtă a Proiectului

Proiectul își propune să analizeze modul în care modificările digitale locale (aplicate strict în regiunile de interes — *ROI*) influențează capacitatea unui detector automat de obiecte (YOLO) de a identifica persoane în imagini termice (infraroșu LWIR). 

Folosind setul de date LLVIP (Kaggle), izolăm corpul uman pe baza bounding box-urilor adnotate și aplicăm tehnici de prelucrare a imaginilor (zgomot Gaussian local, ajustarea statisticilor de intensitate și inpainting Telea). Scopul este evaluarea cantitativă a degradării performanței de detecție ($mAP_{50}$, $Precision$, $Recall$) raportat la calitatea vizuală structurală a imaginii ($SSIM$, $PSNR$).

---

## 1. Analiza Literaturii de Specialitate (State-of-the-Art)

| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu¹ | Tehnologii utilizate² | Metodologie / Abordare³ | Rezultate⁴ | Limitări⁵ | Comentarii suplimentare⁶ |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Wei et al. / 2023 | *Physically Adversarial Infrared Patches With Learnable Shapes and Locations* (CVPR 2023) | Camuflaj digital & atacuri adversariale în imagini termice | PyTorch, YOLOv5, FLIR Dataset, Python, OpenCV | Generarea digitală a unor pete/măști termice (*adversarial patches*) cu formă și poziție optimizate prin gradienți pe regiunea corpului uman pentru suprimarea detecției. | Reducerea ratei de detecție ($Recall$) a detectorului YOLOv5 de la **~88% la sub 15%** folosind modificări locale reduse ca suprafață. | Patch-urile optimizate pot apărea ca pete cu geometrie nenaturală în imagine dacă nu se aplică o regularizare strictă a formei. | Foarte relevant pentru modulul de zgomot/patch pe regiunea bbox-ului; demonstrează că modificarea locală a pixelilor elimină detecția. |
| **2** | You et al. / 2022 | *A Thermal Infrared Pedestrian-Detection Method for Edge Computing Devices* (MDPI Sensors) | Detecția pietonilor în imagini termice (LWIR) | Python, PyTorch, YOLOv5, KAIST & FLIR Datasets, OpenCV | Optimizarea detecției în imagini termice prin ajustarea contrastului local (ROI) și analiza efectului de atenuare a gradientului de temperatură dintre om și fundal. | Obținerea unui $mAP_{50}$ de **84.6%** pe imagini termice nealterate și demonstrarea matematică a scăderii preciziei când contrastul local scade. | Sensibilitate crescută la fundaluri calde (cu temperaturi similare corpului uman) și la imagini cu rezoluție redusă. | Oferă date despre modul în care modificarea histogramei locale/contrastului reduce detecția, fiind util pentru modulul de ajustare a statisticilor. |
| **3** | V. Ganganwar et al. / 2024 | *CamoVeil: Digital Camouflage Pattern Generation System* | Generarea automată a tiparelor de camuflaj digital / Aplicații militare | YOLO, K-means++, CamoGen, OpenCV, filtre de blur (Gaussian, Box, Motion) | Detecția obiectelor cu YOLO, extragerea culorilor dominante cu K-means++, suprapunerea de poligoane colorate în CamoGen, pixelizare și netezirea marginilor. | Generare complet automată a unor tipare de camuflaj adaptate pentru medii naturale (munte, deșert, pădure), fără șabloane preexistente. | Eficiență redusă în medii artificiale/urbane. | Oferă modelul de arhitectură integrată (YOLO + prelucrare locală) pentru izolarea regiunilor de interes și aplicarea camuflajului digital. |
| **4** | X. Jia et al. / 2021 | *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision* | Viziune pe timp de noapte și detecția persoanelor în imagini termice și vizibile | Senzori termici (LWIR) și vizibili aliniați, adnotări în format YOLO | Colectarea și sincronizarea cadrelor duale (RGB-LWIR) în condiții de iluminare scăzută, cu adnotarea manuală/verificată a pietonilor. | Benchmark de 15.486 imagini termice și adnotate, utilizat pe scară largă pentru antrenarea și evaluarea modelelor de detecție a persoanelor. | Datele provin din medii predominant urbane și scenarii nocturne specifice. | Sursă primară de date pentru proiectul nostru (datasetul „Thermal Images for Human Detection”). |
| **5** | ProApto / 2023 | *Thermal Camouflage Systems & Multispectral Materials* | Camuflaj multispectral pentru domeniile vizibil, NIR și LWIR | Materiale textile avansate, structuri sintetice cu emisivitate termică controlată | Proiectarea fizică de echipamente ce reduc contrastul termic dintre persoană și mediul înconjurător prin manipularea emisiilor infraroșii. | Suprimarea vizibilității în camerele termice și perturbarea detecției automate de către senzori. | Eficiența depinde de condițiile de mediu și de temperatura ambientală reală. | Asigură fundamentul fizic și motivația din spatele simulărilor noastre digitale pe imagini termice. |

> **Note explicative ale capului de tabel (PI_P_lab2):**  
> ¹ **Domeniul** în care se aplică soluția descrisă.  
> ² **Tehnologii/echipamente** hardware/software folosite în proiect.  
> ³ **Descrierea metodei** utilizate în proiect.  
> ⁴ **Principalele rezultate** obținute (metrici de evaluare).  
> ⁵ **Limitările** întâmpinate în cercetare sau implementare.  
> ⁶ **Observaţii** privind relevanța pentru proiectul curent.

---

## 2. Arhitectura Soluției și Schema Bloc

Aplicația este proiectată pe o arhitectură modulară, separând citirea datelor, detecția de bază, generarea măștii, algoritmii de prelucrare de imagine și modulul de evaluare statistică.

```text
                                  ┌───────────────────────────┐
                                  │   Dataset Termic LLVIP    │
                                  │ (Imagini + Adnotări YOLO) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   1. Încărcare & Validare Date  │
                               │      (Data Loader & BBox)       │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │ 2. Detecție Referință (Baseline)│
                               │      (Model YOLO pe Original)   │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │    3. Generare Mască ROI        │
                               │  (Mastră binară pe Bounding Box)│
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │ 4. Camuflaj Digital Local (PI)  │
                               │ ├─ Zgomot Gaussian Local        │
                               │ ├─ Ajustare Statistici Locale   │
                               │ └─ Inpainting Telea             │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     5. Re-Detecție Obiecte      │
                               │ (Model YOLO pe Imagine Modific) │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │  6. Evaluare & Export Metrici   │
                               │ ├─ Detecție: mAP50, Recall, IoU │
                               │ └─ Calitate: SSIM_roi, PSNR_roi │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │    7. Interfață Demo (Gradio)   │
                               │ (Afișare Înainte vs. După)      │
                               └─────────────────────────────────┘

### Explicația Componentelor din Schema Bloc

1. **Încărcare & Validare Date** (`src/data/`)
   * Citește imaginile termice și validează adnotările YOLO (verifică coordonatele normalizate și clasa `person`).

2. **Detecție Referință / Baseline** (`src/detection/`)
   * Execută inferența cu modelul YOLOv8 pe imaginea originală pentru a stabili valorile de referință ale detecției.

3. **Generare Mască ROI** (`src/processing/`)
   * Construiește o mască binară cu valoarea `1` în interiorul bounding box-urilor adnotate/detectate și `0` pe fundal.

4. **Camuflaj Digital Local** (`src/processing/`)
   * Aplică algoritmii de prelucrare de imagine (*Zgomot Gaussian*, *Potrivirea statisticilor de intensitate cu fundalul*, *Inpainting Telea*) strict în zona definită de mască.

5. **Re-Detecție Obiecte** (`src/detection/`)
   * Rulează din nou inferența detectorului pe imaginea camuflată.

6. **Evaluare & Export Metrici** (`src/evaluation/`)
   * Compară predicțiile inițiale cu cele noi (`mAP50`, `Recall`, `Precision`, `IoU`) și măsoară calitatea imaginii (`SSIM`, `PSNR`), exportând datele în `results/summary.csv`.

7. **Interfață Demo** (`src/demo/`)
   * Aplicație vizuală interactivă (Gradio / cadru stereo) pentru prezentarea în timp real a comparațiilor de tip „Înainte vs. După”.
