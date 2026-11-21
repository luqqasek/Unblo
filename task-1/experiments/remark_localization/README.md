# ReMark jako sygnał lokalizacji TrustMark — protokół

170 rzeczywistych obrazów TrustMark, Q, 100 bitów bez ECC. Wyłącznie train;
zapisane trzy zewnętrzne i wewnętrzne podziały frozen_extractor/cv3. GPU CUDA
do inference; oficjalny trustmark 0.8.0 oraz niezmienione checkpointy.

Hipoteza: residual po usuwaniu watermarku może wskazywać jego położenie.
Usunięcie wiadomości nie jest dowodem lokalizacji. Fazy ekstrakcji nie uczą wag.

Mapy: R1 średni moduł RGB, R2 norma L2 RGB, R3 moduł residualu po odjęciu
Gaussian sigma=2 px, R4 residual / lokalny kontrast obrazu (sigma=8, floor=1/255)
oraz residual / (lokalna średnia residualu +1/255), R5 RMS kanałów encoder0,
encoder2, bottleneck i ostatniego decoder block. Dodatkowo residual float przed
uint8 jako kontrola kwantyzacji. Mapy do 128×128: torch bilinear antialias,
align_corners=False; Gaussian sigma=0/1/3 komórki, reflect.

Kontrole: blur sigma2, resize 512→256→512 bilinear, gradient obrazu; 24 zdjęcia
COCO val2017 bez celowo nałożonego TrustMark; pary z tym samym obrazem po encode
Q w losowych, wcześniej ustalonych lokalizacjach dla rozmiarów 36/89/162/256,
całego obrazu i dużego watermarku z czystym otworem. Syntetyki wyłącznie do
diagnostyki, nigdy do treningu lub kalibracji rzeczywistej segmentacji.

Dobór progu i mapy osobno w każdym foldzie wyłącznie na zapisanym
inner_validation, bez outer_validation. Kierunek główny: większy residual =
watermark. Progi: 101 kwantyli rozkładu score na inner_validation plus pełna/pusta
maska. Wyniki każdej mapy eksploracyjne; ocena rodziny wybiera mapę i smoothing
wewnątrz foldu. Osobno oracle per-image (kwantyle score, GT tylko do wyboru),
Top-K (K z GT), metryki po odcięciu 4 komórek brzegu (16px obrazu).

Biny powierzchni: <=1%, (1,5%], (5,20%], (20,50%], >50%. Macro AUC/AP per obraz;
dziury to zamknięte komponenty clean wewnątrz FG, niezależne od clean brzegów.

Lokalny ReMark: najpierw znane syntetyczne cropy jako oracle diagnostyczny;
następnie okna bez GT 128/256/384, stride połowa okna, dołączony końcowy offset,
ważona średnia residualu przez Hann z floor0,01. Porównanie skali i średniej skal.

Przed uczeniem integracji wymagamy na rzeczywistych OOF: macro AUC >=0,60,
przewagi AUC >=0,02 nad kontrolą blur/resize/edges wybieraną wewnątrz foldów
i small Top-K lift >=1,5 względem udziału GT. Dodatkowo kontrola par musi wykazać
większy wzrost residualu w regionie watermarku niż poza nim. To bramka rozwojowa,
nie nowy test statystyczny; ewentualna integracja będzie oznaczona jako wybór
na znanych foldach. Brak przejścia bramki oznacza zakończenie bez uczenia głowicy.

Znacząca zmiana wiadomości: >=10/100 zmienionych surowych bitów. Osobno spadek
trafności >=0,10 wśród obrazów o początkowej trafności >=0,75. Nie używamy
`detected=True` przy use_ECC=False jako dowodu detekcji.

Środowisko: `task-1/.venv-rgb-cuda/Scripts/python.exe`. Duże wyniki w `results/`.
Nie zmieniamy istniejących modeli, foldów, zgłoszeń ani aktywnej bazy StegaStamp.
