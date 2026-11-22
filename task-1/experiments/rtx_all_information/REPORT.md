# Wszystkie informacje — eksperyment zatrzymany na żądanie użytkownika

**Brak potwierdzonej poprawy głównej lokalizacji w ukończonych próbach.** PixelSeal i TrustMark zakończyły pełne porównanie 3-fold. StegaStamp zatrzymano podczas przygotowania danych; nie rozpoczął treningu. To wynik dla 340 rzeczywistych obrazów, nie pełne porównanie 510 obrazów.

| Metoda | N | Kontrola mIoU | Wszystkie informacje | Zmiana | 95% bootstrap zmiany |
|---|---:|---:|---:|---:|---|
| pixelseal | 170 | 0.557391 | 0.553402 | -0.003989 | [-0.010468; +0.002335] |
| trustmark | 170 | 0.539056 | 0.535303 | -0.003753 | [-0.010118; +0.002325] |

Oba przedziały obejmują zero. To jeden seed i wielokrotnie używane foldy rozwojowe. Nie ma dowodu poprawy; nie jest to dowód, że żadna architektura nie może skorzystać z dodatkowych informacji.

## Najmniejsze obszary

| Metoda | Typ ≤1% | N | Całkowite pominięcia: kontrola | Całkowite pominięcia: pełny wariant |
|---|---|---:|---:|---:|
| pixelseal | fg_le1 | 17 | 16 | 17 |
| pixelseal | clean_le1 | 12 | 10 | 10 |
| trustmark | fg_le1 | 9 | 8 | 8 |
| trustmark | clean_le1 | 14 | 13 | 14 |

## Czy dodatkowe sygnały zostały nauczone?

| Metoda | Trafność full_watermark na realnym OOF | mIoU pomocniczej maski512 | MAE energii na syntetycznym dev | MAE predykcji zerowej |
|---|---:|---:|---:|---:|
| pixelseal | 0.9765 | 0.552271 | 0.206125 | 0.301122 |
| trustmark | 0.9706 | 0.533162 | 0.242749 | 0.347857 |

Energię oceniono w skali log1p na 200 syntetycznych obrazach rozwojowych na metodę. Lepsze zadania pomocnicze nie przełożyły się na poprawę głównej maski. Ponowna inferencja checkpointów FP32 odtworzyła główne mIoU dokładnie.

## Zakres i sprzęt

Kompaktowy model z zamrożonymi cechami oryginalnego dekodera; nie pełny fine-tuning najlepszego dużego modelu. Każda ukończona metoda: 4000 syntetycznych realizacji treningowych, 200 syntetycznych dev i wszystkie 170 realnych obrazów w trzech foldach. Dwie pełne epoki syntetyczne, 20 epok na części treningowej każdego foldu, seed42, próg0.5. Wszystkie kwalifikujące się realne obrazy odwiedzone w każdej epoce.

Wykorzystane: RGB512, metoda, wszystkie bity wiadomości, oryginalne cechy dekodera i zgodność bitów, maski128/512, granice, odległości i centra składowych, pokrycie, full_watermark oraz dokładne pary syntetyczne czysty/oznaczony. Nazwy i hashe służą identyfikacji i kontroli podziałów. Etykiety i czyste odpowiedniki nie są wejściami przy predykcji. Nie użyto testu do treningu.

RTX3070: szczyt pamięci tensorów trenera 0.500 GB. Łączny czas obu ukończonych porównań modelowych 19.5 min, bez przygotowania danych. Cechy są cache’owane; ta pamięć nie oznacza kosztu pełnego fine-tuningu dużego modelu ani całej pamięci nvidia-smi.

## Pochodzenie danych i zakończenie

Syntetyczne obrazy odtworzono oryginalnymi wrapperami. Hashe RGB różnią się od historycznych w891/4200 przypadków PixelSeal i3043/4200 TrustMark; źródło, wiadomość i maskę zweryfikowano, a cechy policzono na aktualnych pikselach. Oba warianty otrzymały te same odtworzone obrazy. Nie deklarujemy identyczności z historycznymi cache.

StegaStamp: przygotowanie przerwane, bez pomiaru jakości ani treningu. Zachowano ukończone checkpointy, predykcje i częściowy cache. Nie wykonano submission. Procesy przygotowania i koordynatora zakończono oraz sprawdzono ich brak.

Aktualna baza rozwojowa poza tym eksperymentem osiągnęła następnie publiczne0.6677 (#3054). PS/TM pozostały bez zmian względem#2675/#3005; publicznych wyników nie porównujemy bezpośrednio z powyższym CV.

[Protokół](PROTOCOL.md) · [Pomiary](results/measurements.json) · [Diagnostyka](results/auxiliary_diagnostics.json)
