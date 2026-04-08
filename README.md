# Simulazioni di Fisica

Raccolta di simulazioni e visualizzazioni interattive dedicate ai principali argomenti della fisica, sviluppata con finalità didattiche e divulgative.

Il progetto è pensato come un laboratorio digitale consultabile direttamente dal browser, con contenuti che spaziano dai fondamenti della meccanica fino ad ambiti più avanzati come relatività, meccanica quantistica, fisica nucleare e astrofisica.

## Obiettivi del progetto

Questo sito nasce con alcuni obiettivi principali:

- rendere più intuitiva la comprensione dei fenomeni fisici attraverso la visualizzazione dinamica;
- affiancare alla simulazione una spiegazione teorica essenziale ma rigorosa;
- offrire uno strumento utile per studio individuale, didattica scolastica e divulgazione scientifica;
- costruire una raccolta ordinata e progressivamente estendibile di contenuti interattivi.

## Caratteristiche

Le simulazioni presenti nel progetto includono, a seconda dei casi:

- visualizzazione interattiva dei fenomeni fisici;
- modifica in tempo reale dei parametri del modello;
- sezioni teoriche di accompagnamento;
- formulario con le principali relazioni fisiche;
- organizzazione per categorie disciplinari.

L’interfaccia del sito è costruita per permettere una navigazione semplice tra le diverse aree della fisica, con catalogazione dinamica delle simulazioni e suddivisione per livello e argomento.

## Struttura del progetto

Il repository contiene:

- un file `index.html` che funge da pagina principale del catalogo;
- un file `simulations.json` che organizza categorie e simulazioni;
- una serie di simulazioni realizzate principalmente in `HTML`, `CSS` e `JavaScript`;
- in alcuni casi, moduli fisici più complessi sviluppati in `C` e compilati in `WebAssembly (WASM)` per l’esecuzione nel browser.

## Filosofia di sviluppo

Le simulazioni non vogliono essere soltanto “animazioni”, ma strumenti didattici costruiti attorno a un modello fisico leggibile e modificabile.

Per questo motivo, ogni contenuto viene progettato cercando di mantenere un equilibrio tra:

- chiarezza visiva;
- correttezza fisica del modello;
- interattività;
- valore educativo.

Quando il problema fisico è relativamente leggero, l’intera simulazione viene sviluppata in JavaScript.  
Quando invece il carico computazionale diventa più elevato, la parte numerica può essere implementata in C, lasciando al frontend HTML/JavaScript la gestione della visualizzazione e dell’interazione con l’utente.

## Ambiti trattati

Il progetto include o prevede simulazioni nei seguenti ambiti:

- fondamenti e grandezze fisiche;
- meccanica classica;
- gravitazione;
- fluidodinamica;
- termodinamica;
- onde e acustica;
- ottica;
- elettromagnetismo;
- relatività;
- meccanica quantistica;
- fisica nucleare;
- astrofisica.

## Utilizzo

Il sito è pensato per essere aperto direttamente nel browser tramite GitHub Pages.

Per esplorare i contenuti è sufficiente:

1. aprire la pagina principale del progetto;
2. selezionare una categoria;
3. scegliere la simulazione di interesse;
4. modificare i parametri disponibili per osservare il comportamento del sistema.

## Stato del progetto

Il progetto è in continua espansione.  
Accanto alle simulazioni già consolidate, possono essere presenti contenuti in sviluppo o in fase di rifinitura.

## Autore

**Roberto Curcio Rubertini**

## Note finali

Questo repository raccoglie un progetto personale dedicato alla costruzione di un archivio di simulazioni fisiche accessibili, rigorose e didatticamente utili.

L’obiettivo a lungo termine è sviluppare una piattaforma sempre più ampia e organica per la visualizzazione interattiva della fisica.
