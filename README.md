# Metody Inteligencji Obliczeniowej - sem. letni 2020/21
Piotr Matiaszewski, Aleksander Morgała, Jakub Perlak

# Dokumentacja projektu
https://docs.google.com/document/d/1VRB4WF0aJH1gQpLHZBynZ70_gBQOVySSi-H0X3mei68/edit

% Opis plików
1. main_iris.m, main_hab_surv.m, main_seeds.m
Pliki MATLAB zawierające implementacje zaproponowanego w projekcie rozwiązania problemu dopasowania wag reguł FL dzięki PSO dla odpowiednich zbiorów (wyszczególnione w nazwie pliku).
2. generated_FIS.m
Plik MATLAB zawierający automatycznie wygenerowany system rozmyty dla zbioru IRIS oraz HABERMAN'S SURVIVAL.
3. dataset_haberman_survival.txt, dataset_seeds
Pliki z danymi.
4. pliki .mat
Zapisane przestrzenie nazw niektórych wywołań zaproponowanego rozwiązania. W nazwie pliku określone są:
-> nazwa zbioru dla którego rozwiazanie było wywołane;
-> 'ws' od ang. WorkSpace;
-> wartości parametrów SelfAdjustmentWeight oraz SocialAdjustmentWeight dla których zostało wywołane PSO.