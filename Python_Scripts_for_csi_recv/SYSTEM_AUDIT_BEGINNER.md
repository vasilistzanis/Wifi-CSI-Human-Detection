# Οδηγός και τεχνικό audit του συστήματος

Αυτό το έγγραφο εξηγεί το σύστημα από την αρχή, με απλή γλώσσα και με βάση
τον κώδικα που υπάρχει σήμερα. Σκοπός του είναι να μπορείς να ακολουθήσεις τη
διαδρομή από το ESP32 μέχρι την πρόβλεψη στην οθόνη και να ξέρεις ποιος
υπολογιστής κάνει τι.

Είναι τεχνική περιήγηση αρχιτεκτονικής και κώδικα, όχι πλήρης έλεγχος
διείσδυσης ή πιστοποίηση production.

> **Σημαντικό:** Περιγράφεται η τωρινή υλοποίηση. Δεν χρησιμοποιείται MQTT.
> Το desktop επικοινωνεί με το API μέσω HTTP/HTTPS και το website λαμβάνει
> ζωντανές ενημερώσεις από το API μέσω WebSocket.

## 1. Η ιδέα με μία ματιά

Το σύστημα έχει τρία βασικά μέρη:

1. **ESP32:** μετρά τα Wi-Fi CSI δεδομένα και τα στέλνει μέσω USB/σειριακής
   σύνδεσης στον υπολογιστή.
2. **Desktop εφαρμογή:** διαβάζει τη σειριακή θύρα, δείχνει τοπικά γραφήματα
   και προωθεί τα αρχικά CSI frames στον API server.
3. **API server και website:** ο server κάνει το inference, κρατά ρυθμίσεις και
   ιστορικό, και το website εμφανίζει την κατάσταση και τις προβλέψεις.

Όταν το desktop είναι συνδεδεμένο στο API, το API είναι η **μοναδική πηγή
προβλέψεων**. Το desktop δεν τρέχει δεύτερο μοντέλο παράλληλα. Έτσι αποφεύγεται
να βλέπεις διαφορετικό αποτέλεσμα στο desktop και διαφορετικό στο website.

```text
 ESP32
   │ CSI_DATA μέσω USB / σειριακής θύρας
   ▼
 Υπολογιστής με live_dashboard.py
   ├─ εμφανίζει τοπικά σήμα και γραφήματα
   ├─ κρατά προσωρινή ουρά CSI frames
   ├─ στέλνει batches με HTTP/HTTPS ──────────────┐
   ├─ στέλνει αλλαγές ρυθμίσεων με HTTP/HTTPS ──┤
   └─ ζητά περιοδικά status με HTTP/HTTPS ──────┤
                                                 ▼
                                      FastAPI API server
                                      ├─ ελέγχει και αναλύει frames
                                      ├─ κάνει inference
                                      ├─ αποθηκεύει δεδομένα σε SQLite
                                      └─ στέλνει live events
                                                 │ WebSocket
                                                 ▼
                                      Browser / website dashboard

 Σε δημόσια εγκατάσταση:
 Desktop ── HTTPS ──► Nginx / TLS ── HTTP τοπικά ──► API container
```

Το API ξεκινά από το `web_app.server:app`. Οι αρμοδιότητές του είναι χωρισμένες
στα [routes.py](./web_app/routes.py), [inference.py](./web_app/inference.py),
[persistence.py](./web_app/persistence.py) και [schemas.py](./web_app/schemas.py).
Τα runnable desktop/collector προγράμματα υλοποιούνται στο [apps/](./apps),
ενώ τα benchmark και research experiment εργαλεία είναι στο
[experiments/](./experiments). Τα μικρά root-level scripts, όπως το
[live_dashboard.py](./live_dashboard.py), είναι compatibility launchers για να
συνεχίσουν να λειτουργούν οι παλιές εντολές. Το κύριο παράθυρο και ο
συντονισμός σελίδων είναι
στο [apps/desktop/dashboard_window.py](./apps/desktop/dashboard_window.py), οι
οθόνες στο [apps/desktop/dashboard_pages.py](./apps/desktop/dashboard_pages.py),
η κύρια οθόνη Monitor στο
[apps/desktop/dashboard_monitor.py](./apps/desktop/dashboard_monitor.py), και
οι ρυθμίσεις στο
[apps/desktop/dashboard_settings.py](./apps/desktop/dashboard_settings.py). Τα
επαναχρησιμοποιήσιμα widgets είναι στο
[apps/desktop/dashboard_widgets.py](./apps/desktop/dashboard_widgets.py). Ο reader που
διαχειρίζεται τη σειριακή σύνδεση και τα shared-memory snapshots βρίσκεται
στο [apps/desktop/desktop_reader.py](./apps/desktop/desktop_reader.py). Η φόρτωση desktop μοντέλων
βρίσκεται στο [apps/desktop/desktop_models.py](./apps/desktop/desktop_models.py). Οι κοινές λειτουργίες
edge είναι στα [csi_serial.py](./csi_serial.py) και
[csi_forwarding.py](./csi_forwarding.py). Το website ξεκινά από το
[app.js](./web_app/static/app.js).

## 2. Λεξιλόγιο για αρχάριο

| Όρος | Απλή εξήγηση |
|---|---|
| **Client** | Πρόγραμμα που ζητά υπηρεσία. Εδώ client είναι το desktop ή ο browser. |
| **Server** | Πρόγραμμα που περιμένει αιτήματα και απαντά. Εδώ είναι το FastAPI. |
| **API** | Συμφωνημένοι δρόμοι/διευθύνσεις και μορφές δεδομένων για επικοινωνία. |
| **HTTP** | Ο client στέλνει ένα αίτημα και ο server απαντά. |
| **HTTPS** | HTTP μέσα σε κρυπτογραφημένη σύνδεση TLS. Χρειάζεται για σύνδεση μέσω Internet. |
| **WebSocket** | Ανοιχτή αμφίδρομη σύνδεση για συνεχή ενημέρωση, χωρίς επαναλαμβανόμενα HTTP αιτήματα. |
| **Inference** | Χρήση του ήδη εκπαιδευμένου μοντέλου για πρόβλεψη σε νέα δεδομένα. |
| **Batch** | Πολλά frames σε ένα αίτημα αντί για ένα ξεχωριστό αίτημα ανά frame. |
| **Reverse proxy** | Ενδιάμεσος server, π.χ. Nginx, που δέχεται δημόσια κίνηση και την προωθεί στο API. |
| **TLS certificate** | Ψηφιακό πιστοποιητικό που επιτρέπει στον browser να επιβεβαιώσει και να κρυπτογραφήσει το HTTPS site. |
| **Docker image** | Το πακέτο με τον κώδικα και τις εξαρτήσεις που χρειάζεται για να ξεκινήσει το container. |
| **Container** | Απομονωμένη διεργασία που τρέχει από Docker image. |
| **Volume** | Μόνιμος χώρος αποθήκευσης που επιβιώνει όταν σταματήσει ή ξαναφτιαχτεί το container. |
| **Git LFS** | Μηχανισμός Git για μεγάλα αρχεία, όπως τα εκπαιδευμένα μοντέλα. |

## 3. Τι κάνει κάθε αρχείο

| Αρχείο | Ρόλος |
|---|---|
| [apps/live_dashboard.py](./apps/live_dashboard.py) | Υλοποίηση CLI, αρχικοποίηση και εκκίνηση του desktop dashboard. |
| [live_dashboard.py](./live_dashboard.py) | Compatibility launcher για την παλιά εντολή εκκίνησης. |
| [apps/desktop/dashboard_window.py](./apps/desktop/dashboard_window.py) | Κύριο παράθυρο, συντονισμός runtime και αλλαγή σελίδων. |
| [apps/desktop/dashboard_monitor.py](./apps/desktop/dashboard_monitor.py) | Η ζωντανή σελίδα σημάτων και προβλέψεων Monitor. |
| [apps/desktop/dashboard_pages.py](./apps/desktop/dashboard_pages.py) | Οι σελίδες Signal View, Activity Log και System Info. |
| [apps/desktop/dashboard_settings.py](./apps/desktop/dashboard_settings.py) | Η επιλογή μοντέλου, mode και runtime ρυθμίσεων. |
| [apps/desktop/dashboard_widgets.py](./apps/desktop/dashboard_widgets.py) | Κοινά widgets, sidebar και helpers για τις σελίδες. |
| [apps/desktop/desktop_reader.py](./apps/desktop/desktop_reader.py) | Reader process, CSI serial/demo ροή και shared-memory snapshots. |
| [apps/desktop/desktop_models.py](./apps/desktop/desktop_models.py) | Φόρτωση των εκπαιδευμένων desktop model artifacts. |
| [csi_forwarding.py](./csi_forwarding.py) | Κοινός HTTP client: batches, αλλαγές μοντέλου/ρυθμίσεων, ανάγνωση status και retries. |
| [csi_serial.py](./csi_serial.py) | Κοινή ανάγνωση σειριακών γραμμών CSI, που χρησιμοποιούν το desktop και ο CLI collector. |
| [apps/desktop/dashboard_theme.py](./apps/desktop/dashboard_theme.py) | Χρώματα, ονόματα κλάσεων/μοντέλων και κοινά stylesheet στοιχεία του desktop. |
| [apps/desktop/desktop_inference.py](./apps/desktop/desktop_inference.py) | Child process και queue lifecycle για τοπικό inference του desktop όταν δεν χρησιμοποιείται API. |
| [csi_parser.py](./csi_parser.py) | Μετατροπή της γραμμής κειμένου `CSI_DATA` σε αριθμητικό CSI frame. |
| [server.py](./web_app/server.py) | Δημιουργεί το FastAPI app, κάνει startup/shutdown του inference engine και ρυθμίζει middleware, static αρχεία και health checks. |
| [routes.py](./web_app/routes.py) | HTTP/WebSocket API routes, authentication dependencies και έλεγχος των edge requests. |
| [inference.py](./web_app/inference.py) | Φόρτωση μοντέλων, CSI batch processing, predictions και runtime inference settings. |
| [persistence.py](./web_app/persistence.py) | SQLite schema/queries, runtime settings, password hashing, login sessions και bootstrap admin. |
| [schemas.py](./web_app/schemas.py) | Τύποι και όρια των JSON αιτημάτων που δέχεται το API. |
| [apps/live_predict.py](./apps/live_predict.py) | Terminal inference εφαρμογή και κοινή συνάρτηση inference που χρησιμοποιεί το API. |
| [live_predict.py](./live_predict.py) | Compatibility launcher και διατήρηση του import path που χρησιμοποιεί το API. |
| [apps/csi_logger.py](./apps/csi_logger.py) | Εφαρμογή καταγραφής CSI frames από τη σειριακή θύρα. |
| [experiments/benchmark_latency.py](./experiments/benchmark_latency.py) | Εργαλείο benchmark καθυστέρησης μοντέλων. |
| [experiments/experiment_runner.py](./experiments/experiment_runner.py) | Εκτέλεση cross-subject, cross-environment και ablation πειραμάτων. |
| [app.js](./web_app/static/app.js) | Login και browser συμπεριφορά: HTTP αιτήματα για ιστορικό και WebSocket για live ενημερώσεις. |
| [index.html](./web_app/static/index.html) | Δομή/στοιχεία του website dashboard. |
| [app.css](./web_app/static/app.css) | Εμφάνιση του website dashboard. |
| [start_api.sh](./start_api.sh) | Τοπικός launcher του API μέσα στο Python virtual environment. |
| [start_dashboard.sh](./start_dashboard.sh) | Τοπικός launcher του desktop και σύνδεσή του με το API. |
| [collector.py](./web_app/collector.py) | Εναλλακτικός command-line collector για υπολογιστή με ESP32. Δεν είναι το ίδιο πρόγραμμα με το γραφικό dashboard. |
| [Dockerfile](./web_app/Dockerfile) | Οδηγίες για την κατασκευή της εικόνας του API. |
| [docker-compose.yml](./web_app/docker-compose.yml) | Ρυθμίσεις εκκίνησης του container, mounts, port, health check και restart. |
| [nginx.conf.example](./web_app/nginx.conf.example) | Παράδειγμα Nginx reverse proxy για HTTPS και WebSocket. Δεν είναι αυτόματη εγκατάσταση Nginx ή πιστοποιητικών. |
| [requirements.txt](./web_app/requirements.txt) | Άμεσες εξαρτήσεις του API. |
| [requirements.lock.txt](./web_app/requirements.lock.txt) | Κλειδωμένες εκδόσεις εξαρτήσεων που χρησιμοποιεί το Docker build. |
| [.dockerignore](./.dockerignore) | Αποκλείει από το Docker build context δεδομένα, μυστικά και μεγάλα model artifacts. |
| [.env.example](./web_app/.env.example) | Πρότυπο μεταβλητών περιβάλλοντος. Δεν περιέχει τις πραγματικές τιμές που πρέπει να χρησιμοποιούνται. |

### Πού εκπαιδεύεται το μοντέλο;

Το API **δεν εκπαιδεύει** μοντέλα. Η εκπαίδευση γίνεται χωριστά από τα
scripts του project, όπως το `csi_ml_pipeline.py`, και παράγει artifacts
(pipeline, label encoder και classifier). Το API φορτώνει αυτά τα artifacts
και τα χρησιμοποιεί για inference σε νέα frames. Με άλλα λόγια:

```text
Εκπαίδευση εκτός server ──► αποθήκευση model artifacts
                                      │
                                      ▼
Server startup ──► φόρτωση artifacts ──► inference σε live CSI frames
```

Η αλλαγή classifier στο dashboard αλλάζει ποιο ήδη εκπαιδευμένο artifact
χρησιμοποιείται. Δεν ξεκινά νέα εκπαίδευση.

## 4. Τι γίνεται όταν ξεκινά τοπικά

### 4.1 Πρώτα ξεκινά το API

Από τον φάκελο `Python_Scripts_for_csi_recv` τρέχεις:

```bash
./start_api.sh
```

Ο launcher:

1. Βρίσκει τον φάκελο του project.
2. Διαβάζει το `web_app/.env`.
3. Ελέγχει ότι το edge token έχει τουλάχιστον 32 χαρακτήρες και υπάρχει όνομα admin.
4. Ελέγχει ότι υπάρχει το `.venv`.
5. Ορίζει την τοπική διαδρομή της SQLite, αν δεν έχει ήδη οριστεί.
6. Ξεκινά το Uvicorn με το FastAPI app στο `127.0.0.1:8000`.

Το `127.0.0.1` σημαίνει «μόνο αυτός ο υπολογιστής». Ο τοπικός launcher δεν
ανοίγει το API κατευθείαν στο Internet.

Κατά την εκκίνηση, το `lifespan` του FastAPI:

1. ελέγχει τις βασικές ρυθμίσεις,
2. δημιουργεί/ενημερώνει τους πίνακες SQLite,
3. δημιουργεί τον πρώτο admin μόνο αν δεν υπάρχει ήδη,
4. φορτώνει το επιλεγμένο pipeline και μοντέλο,
5. ξεκινά background watchdog που κλείνει activity sessions μετά από αδράνεια.

Αν λείπει το επιλεγμένο μοντέλο ή οι ρυθμίσεις δεν είναι έγκυρες, η εκκίνηση
αποτυγχάνει με σφάλμα αντί να προσποιηθεί ότι το API είναι έτοιμο.

### 4.2 Μετά ξεκινά το desktop

Σε δεύτερο terminal:

```bash
./start_dashboard.sh
```

Ο launcher φορτώνει το ίδιο `.env`, χρησιμοποιεί από προεπιλογή API URL
`http://127.0.0.1:8000`, ελέγχει το `/readyz` και έπειτα ανοίγει το
`live_dashboard.py`. Η σειριακή θύρα προεπιλογής είναι `/dev/ttyACM0`, αλλά
μπορεί να αλλάξει με `CSI_SERIAL_PORT` ή με τα arguments του dashboard.

Αν ο server δεν είναι έτοιμος, ο launcher σταματά με μήνυμα. Αν το API δεν
ρυθμιστεί στο ίδιο το dashboard, εκείνο μπορεί να λειτουργήσει σε offline
mode, με τοπικό inference, εφόσον υπάρχουν τα μοντέλα. Σε demo mode
χρησιμοποιεί συνθετικά δεδομένα και δεν αντιπροσωπεύει πραγματική μέτρηση.

## 5. Η διαδρομή ενός frame: από το ESP32 μέχρι την πρόβλεψη

### Βήμα 1 — Σειριακή ανάγνωση

Το ESP32 στέλνει γραμμές που αρχίζουν με `CSI_DATA`. Ο reader του desktop
διαβάζει τη σειριακή θύρα. Για κάθε γραμμή:

- αν δεν είναι CSI γραμμή, την αγνοεί,
- αν είναι CSI γραμμή, κρατά την αρχική γραμμή για αποστολή στο API,
- προσπαθεί επίσης να την αναλύσει τοπικά για τα γραφήματα.

Το desktop έχει δικό του process για την ανάγνωση και ενημέρωση των γραφημάτων,
ώστε η ανάγνωση της σειριακής θύρας να μην εξαρτάται από το UI thread.

### Βήμα 2 — Προσωρινή ουρά και batching

Τα frames μπαίνουν σε bounded ουρά μνήμης. Ένα background thread συγκεντρώνει
έως 10 frames ή περιμένει περίπου 100 ms και μετά τα στέλνει σε ένα αίτημα.
Έτσι δεν δημιουργείται HTTP αίτημα για κάθε μεμονωμένο CSI frame.

Το αίτημα πηγαίνει στο:

```text
POST /api/ingest
Authorization header με το μυστικό token συσκευής
Content-Type: application/json
```

Το σώμα έχει περίπου αυτή τη μορφή:

```json
{
  "frames": [
    "CSI_DATA,...",
    "CSI_DATA,..."
  ]
}
```

Η πραγματική γραμμή CSI μπορεί να είναι μεγάλη· το παραπάνω είναι μόνο
σχηματικό παράδειγμα.

### Βήμα 3 — Έλεγχοι στο API

Πριν δεχτεί frames, το endpoint ελέγχει:

- ότι το αίτημα έχει έγκυρο edge token συσκευής,
- ότι το batch περιέχει από 1 έως 100 γραμμές,
- ότι καμία γραμμή δεν ξεπερνά τους 16.384 χαρακτήρες που ορίζει ο κώδικας.

Έπειτα το `InferenceEngine.process_batch()` περνά κάθε γραμμή στον κοινό
`parse_csi_line()` parser. Frames που δεν αναλύονται ή έχουν λάθος αριθμό
subcarriers μετρώνται ως invalid και δεν μπαίνουν στο inference buffer.

### Βήμα 4 — Buffer, preprocessing και μοντέλο

Για έγκυρα frames, ο server:

1. τα βάζει σε rolling buffer,
2. περιμένει να γεμίσει το απαιτούμενο window και warm-up context,
3. περιμένει τον αριθμό νέων frames που ορίζει το `step_size`,
4. καλεί την κοινή `run_inference()` του `live_predict.py`.

Η συνάρτηση inference, σε γενικές γραμμές:

1. φτιάχνει πίνακα από τα complex CSI frames,
2. εφαρμόζει το αποθηκευμένο `CSIPipeline` preprocessing,
3. μετατρέπει το processed window σε feature vector,
4. περνά τα χαρακτηριστικά στο classifier,
5. παίρνει την προβλεπόμενη κλάση και πιθανότητες.

Στη λειτουργία HAR, το API χρησιμοποιεί EMA (εξομάλυνση πιθανοτήτων),
confidence threshold και hysteresis για να μη μεταπηδά η ετικέτα με κάθε
μικρή διακύμανση. Στη λειτουργία Motion, συνδυάζει την έξοδο του μοντέλου με
εκτίμηση motion intensity/threshold.

Οι αριθμητικές τιμές αυτών των ρυθμίσεων αποθηκεύονται από το API. Το ακριβές
αποτέλεσμα πάντα εξαρτάται από τα εκπαιδευμένα artifacts, τις ρυθμίσεις και
την ποιότητα/συμβατότητα CSI εισόδου.

### Βήμα 5 — Απόκριση και ενημέρωση όλων των οθονών

Το HTTP response του `/api/ingest` περιλαμβάνει πόσα frames έγιναν δεκτά και
την τρέχουσα κατάσταση. Ταυτόχρονα, το API εκπέμπει WebSocket event στους
συνδεδεμένους browsers. Το desktop ζητά επίσης το `/api/edge/status` περιοδικά
για να εμφανίζει την authoritative κατάσταση του server.

**Άρα:** τα γραφήματα σήματος στο desktop μπορούν να ανανεώνονται τοπικά,
αλλά η πρόβλεψη που εμφανίζεται σε API mode προέρχεται από το API.

## 6. Πώς περνούν οι ρυθμίσεις

Από τη σελίδα Settings του desktop μπορείς να αλλάξεις μοντέλο, mode και
runtime παραμέτρους. Το desktop στέλνει τις αλλαγές στο API με HTTP:

- `POST /api/edge/model` για επιλογή μοντέλου,
- `POST /api/edge/settings` για mode και λοιπές runtime ρυθμίσεις.

Τα edge endpoints ελέγχουν το ίδιο μυστικό token συσκευής. Το Pydantic
ελέγχει τύπους και όρια τιμών πριν ο server δεχτεί την αλλαγή. Όταν αλλάζει μοντέλο ή mode,
πρώτα φορτώνονται τα απαραίτητα artifacts. Αν αυτό αποτύχει, η αλλαγή
απορρίπτεται. Στις έγκυρες αλλαγές ο server:

1. ενημερώνει την κατάσταση του inference engine,
2. αποθηκεύει τις τιμές στον πίνακα `runtime_settings`,
3. καταγράφει την αλλαγή για το session history,
4. εκπέμπει `settings_changed` event προς τους browsers.

Με το επόμενο restart, τα αποθηκευμένα runtime settings διαβάζονται ξανά από
τη βάση. Η ιστοσελίδα είναι κυρίως read-only mirror: δείχνει τις ρυθμίσεις,
αλλά η ροή αλλαγής ρυθμίσεων γίνεται από το desktop.

### API διαδρομές σε απλά λόγια

| Διαδρομή | Μέθοδος | Ποιος τη χρησιμοποιεί | Τι κάνει |
|---|---|---|---|
| `/` και `/static/...` | GET | Browser | Στέλνει το website HTML, JavaScript και CSS. |
| `/healthz` | GET | Health check | Ελέγχει αν απαντά η HTTP εφαρμογή. |
| `/readyz` | GET | Launcher / Docker health check | Ελέγχει αν είναι διαθέσιμο το inference engine. |
| `/api/auth/login` | POST | Browser | Ελέγχει credentials και δημιουργεί session cookie. |
| `/api/auth/me` | GET | Browser | Ελέγχει αν το browser session παραμένει έγκυρο. |
| `/api/auth/logout` | POST | Browser | Διαγράφει το session. |
| `/api/status`, `/api/models` | GET | Authenticated API client | Επιστρέφει status και διαθέσιμα μοντέλα. |
| `/api/predictions` | GET | Website / authenticated API client | Φέρνει πρόσφατο αποθηκευμένο prediction history. |
| `/api/activity-sessions` | GET | Website / authenticated API client | Φέρνει λίστα sessions. |
| `/api/activity-sessions/{id}` | GET | Website / authenticated API client | Φέρνει λεπτομέρειες ενός session. |
| `/api/settings/model` | POST | Authenticated API client | Αλλάζει μοντέλο μέσω user session. |
| `/api/edge/status` | GET | Desktop | Διαβάζει authoritative status με edge token. |
| `/api/edge/settings` | POST | Desktop | Αλλάζει runtime settings με edge token. |
| `/api/edge/model` | POST | Desktop | Επιλέγει μοντέλο με edge token. |
| `/api/ingest` | POST | Desktop / collector | Στέλνει CSI batch με edge token. |
| `/api/events` | WebSocket | Συνδεδεμένος browser | Στέλνει live status, predictions και events. |

Σημείωση: `/api/edge/...` δεν σημαίνει δημόσιο χωρίς προστασία. Τα routes
αυτά ελέγχουν το token της συσκευής. Αντίστοιχα, τα `/api/...` ιστορικού και
dashboard ελέγχονται με το browser session cookie. Το website της τωρινής
έκδοσης χρησιμοποιεί κυρίως το WebSocket για live δεδομένα και HTTP για
sessions/history· η διαδρομή `/api/settings/model` υπάρχει στο API, αλλά η
τρέχουσα ροή αλλαγών ρυθμίσεων γίνεται από το desktop.

## 7. HTTP, HTTPS και WebSocket: ποιο κάνει τι

| Σύνδεση | Χρήση |
|---|---|
| Desktop → API, HTTP POST | Αποστολή batch CSI frames. |
| Desktop → API, HTTP POST | Αλλαγή μοντέλου και inference settings. |
| Desktop → API, HTTP GET | Ανάκτηση authoritative status. |
| Browser → API, HTTP | Login, status, μοντέλα, ιστορικό και sessions. |
| Browser ↔ API, WebSocket | Ζωντανά status, προβλέψεις και αλλαγές sessions/settings. |
| Desktop → δημόσιος API, HTTPS | Κρυπτογραφημένη μεταφορά μέσω Internet. |

Το WebSocket ξεκινά ως HTTP handshake και μετά παραμένει ανοιχτό. Σε site με
HTTPS, ο browser χρησιμοποιεί `wss://` (WebSocket με TLS). Ο κώδικας του
website επανασυνδέεται όταν η σύνδεση κλείσει, εκτός αν χρειάζεται ξανά login.

**Δεν χρησιμοποιείται MQTT στην τωρινή έκδοση.** Δεν υπάρχει broker, MQTT
client, MQTT topic ή MQTT authentication setup στο deployment.

## 8. Login και ασφάλεια πρόσβασης

Υπάρχουν δύο διαφορετικοί τρόποι πιστοποίησης:

### Browser / άνθρωπος

- Ο χρήστης συνδέεται με username και password.
- Ο κωδικός δεν αποθηκεύεται αυτούσιος: χρησιμοποιείται PBKDF2-HMAC-SHA256
  με salt και 600.000 iterations.
- Με επιτυχημένο login εκδίδεται τυχαίο session cookie.
- Στη βάση αποθηκεύεται hash του session token, όχι το token αυτούσιο.
- Το cookie είναι `HttpOnly`, `SameSite=Strict` και λήγει σε 12 ώρες.
- Σε production πρέπει να είναι ενεργό `CSI_COOKIE_SECURE=true`, ώστε το
  cookie να στέλνεται μόνο μέσω HTTPS.
- Δεν υπάρχει δημόσιο self-signup. Ο admin λογαριασμός δημιουργείται στην
  πρώτη εκκίνηση αν δεν υπάρχει ήδη. Αλλαγή του env password από μόνη της δεν
  αντικαθιστά password ήδη υπάρχοντος χρήστη.

### Desktop / edge device

- Το desktop έχει κοινό μυστικό `CSI_EDGE_TOKEN`.
- Το τοποθετεί στο HTTP Authorization header.
- Ο server το συγκρίνει με constant-time comparison.
- Το token δεν πρέπει να μπαίνει σε Git, screenshot, δημόσιο log ή δημόσιο
  μήνυμα. Αν εκτεθεί, πρέπει να αντικατασταθεί και στις δύο πλευρές.

Επιπλέον, το API:

- περιορίζει επιτρεπόμενα Host headers με `CSI_ALLOWED_HOSTS`,
- ελέγχει origin του WebSocket με `CSI_ALLOWED_ORIGINS`,
- προσθέτει βασικά security headers και Content Security Policy,
- περιορίζει αποτυχημένα login attempts στη διεργασία,
- έχει επίσης login rate limit στο Nginx example,
- αφήνει το API documentation απενεργοποιημένο εκ προεπιλογής.

Τα security headers και οι έλεγχοι είναι χρήσιμα επίπεδα άμυνας· δεν
αντικαθιστούν TLS, σωστά secrets, firewall, ενημερώσεις και backups.

## 9. Τι αποθηκεύεται στη SQLite

Η SQLite βάση κρατά εφαρμογικά δεδομένα. Στο Docker, η διαδρομή είναι
`/data/app.sqlite3`, η οποία συνδέεται με persistent Docker volume. Σε local
run, η προεπιλεγμένη βάση είναι κάτω από `web_app/data/`.

| Πίνακας | Τι κρατά |
|---|---|
| `users` | Όνομα χρήστη, salt και password hash. |
| `sessions` | Hash session token, username και λήξη. |
| `runtime_settings` | Τις τελευταίες ενεργές ρυθμίσεις inference. |
| `activity_sessions` | Χρονικά όρια και σύνολο frames ανά συνεδρία λειτουργίας. |
| `activity_stats` | Αθροιστικές/ομαδοποιημένες στατιστικές ανά session, ετικέτα και μοντέλο. |
| `activity_settings_changes` | Ιστορικό αλλαγών ρυθμίσεων. |
| `predictions` | Δείγμα/ιστορικό προβλέψεων με confidence, πιθανότητες και timestamps. |

Η βάση έχει ενεργό SQLite WAL mode, κατάλληλο για concurrent reads/writes σε
μια μικρή εφαρμογή. Τα raw CSI frames **δεν αποθηκεύονται μόνιμα**·
επεξεργάζονται και κρατιούνται προσωρινά στον buffer μνήμης του API. Η εφαρμογή
κρατά έως 50.000 εγγραφές στον πίνακα predictions και διατηρεί activity
summaries.

### Σημαντική λεπτομέρεια για τα counts

Το code path αποθηκεύει activity statistic όταν αλλάξει η προβλεπόμενη
ετικέτα ή περάσουν τουλάχιστον 4 δευτερόλεπτα από την προηγούμενη statistic
εγγραφή. Η εγγραφή στη λίστα `predictions` γίνεται το πολύ ανά 5
δευτερόλεπτα. Συνεπώς:

- οι αριθμοί αυτοί δεν είναι απαραίτητα μέτρηση κάθε εσωτερικού inference
  βήματος,
- το activity history είναι κατάλληλο για σύνοψη/παρακολούθηση, όχι για
  πλήρες, αδιάλειπτο raw audit log κάθε πρόβλεψης.

Νέα activity session ξεκινά όταν φτάσουν frames μετά από κενό πάνω από 15
δευτερόλεπτα. Το watchdog κλείνει το ενεργό session όταν δεν φτάνουν νέα frames.

## 10. Docker: τι κάνει κάθε κομμάτι

Το [Dockerfile](./web_app/Dockerfile) κατασκευάζει ένα API image:

1. Ξεκινά από `python:3.12-slim`.
2. Εγκαθιστά τις εκδόσεις από το `requirements.lock.txt`.
3. Δημιουργεί μη-root χρήστη `app`.
4. Αντιγράφει τον κώδικα στο image.
5. Εκκινεί Uvicorn στο port 8000 μέσα στο container.

Το `EXPOSE 8000` στο Dockerfile είναι δήλωση/τεκμηρίωση της θύρας του
container· από μόνο του δεν ανοίγει δημόσια θύρα στο Internet.

Το [docker-compose.yml](./web_app/docker-compose.yml):

- περνά τις ρυθμίσεις από το `web_app/.env`,
- δημοσιεύει το API μόνο στο `127.0.0.1:8000` του host,
- συνδέει το `csi_app_data` persistent volume στο `/data`,
- κάνει read-only mount τα HAR και Motion model directories,
- κάνει το filesystem του container read-only, εκτός από προσωρινό `/tmp`
  και το data volume,
- αφαιρεί Linux capabilities και ορίζει `no-new-privileges`,
- χρησιμοποιεί `restart: unless-stopped`,
- ελέγχει το `/readyz` περιοδικά.

Το `.dockerignore` αποκλείει το `.env`, τα datasets, το virtual environment και
τα μεγάλα model files από το build context. Τα μοντέλα παρέχονται χωριστά με
read-only volume mounts. Γι' αυτό πρέπει να υπάρχουν σωστά στον host πριν
ξεκινήσει το container.

Ενδεικτική εκκίνηση, από τον φάκελο `Python_Scripts_for_csi_recv`:

```bash
docker compose -f web_app/docker-compose.yml up --build -d
```

Χρήσιμες εντολές:

```bash
docker compose -f web_app/docker-compose.yml ps
docker compose -f web_app/docker-compose.yml logs -f api
docker compose -f web_app/docker-compose.yml restart api
```

Τοπικός έλεγχος:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

### `healthz` έναντι `readyz`

- `/healthz`: δείχνει ότι η HTTP εφαρμογή απαντά.
- `/readyz`: εξαρτάται από το φορτωμένο inference engine και επιστρέφει το
  ενεργό μοντέλο.

Η επιτυχία του `/healthz` μόνη της δεν αποδεικνύει ότι όλη η ροή CSI ή η
πραγματική συσκευή λειτουργεί.

## 11. Nginx και HTTPS σε πραγματικό server

Σε δημόσια εγκατάσταση, ο browser και το desktop δεν πρέπει να συνδέονται σε
απλό HTTP πάνω από Internet. Το προτεινόμενο μονοπάτι είναι:

```text
Browser / Desktop
       │ HTTPS ή WSS
       ▼
Nginx στο δημόσιο 443
       │ HTTP τοπικά μέσω loopback
       ▼
API στο 127.0.0.1:8000
```

Το [nginx.conf.example](./web_app/nginx.conf.example) δείχνει:

- listener TLS στο 443,
- σύγχρονα TLS protocols,
- certificate και private key paths,
- περιορισμό του μέγιστου HTTP request body,
- προώθηση Host, client IP και HTTPS scheme headers,
- τα WebSocket `Upgrade`/`Connection` headers,
- μεγαλύτερο proxy read timeout για τις ανοιχτές WebSocket συνδέσεις,
- πρόσθετο rate limit στο login endpoint.

Το αρχείο είναι **παράδειγμα**, όχι έτοιμο configuration για copy/paste:

- πρέπει να αντικατασταθεί το `csi.example.com`,
- πρέπει να εκδοθούν και να εγκατασταθούν έγκυρα TLS certificates,
- το Nginx πρέπει να είναι ενεργό και να έχει ανοιχτές τις θύρες 80/443,
- πρέπει να ρυθμιστούν σωστά DNS και firewall,
- πρέπει να ελεγχθεί με `nginx -t` πριν γίνει reload.

Το example περιέχει listener 443, αλλά δεν ορίζει από μόνο του listener στη
θύρα 80 για ανακατεύθυνση σε HTTPS ή διαδικασία έκδοσης/ανανέωσης
πιστοποιητικών. Αυτά χρειάζονται επιπλέον ρύθμιση στο deployment. Το
HSTS header καλό είναι να ενεργοποιείται αφού έχει επιβεβαιωθεί ότι το HTTPS
λειτουργεί σωστά για το domain.

Τα `CSI_ALLOWED_HOSTS` και `CSI_ALLOWED_ORIGINS` πρέπει επίσης να συμφωνούν με
το domain:

- το `CSI_ALLOWED_HOSTS` θέλει hostname χωρίς scheme,
- το `CSI_ALLOWED_ORIGINS` θέλει ολόκληρο origin, π.χ. `https://csi.example.com`.

### Βασικές μεταβλητές περιβάλλοντος

| Μεταβλητή | Τι ρυθμίζει |
|---|---|
| `CSI_ADMIN_USERNAME` | Username πρώτου admin κατά το αρχικό bootstrap. |
| `CSI_ADMIN_PASSWORD` | Password πρώτου admin· χρησιμοποιείται μόνο αν ο χρήστης δεν υπάρχει ήδη. |
| `CSI_EDGE_TOKEN` | Κοινό μυστικό για τα edge routes. Πρέπει να υπάρχει τόσο στον API host όσο και στο desktop. |
| `CSI_SERVER_URL` | URL του API που χρησιμοποιεί το desktop, τοπικό ή HTTPS domain. |
| `CSI_MODEL` / `CSI_MODE` | Αρχική επιλογή classifier και λειτουργίας, αν δεν υπάρχουν ήδη αποθηκευμένα runtime settings. |
| `CSI_MODEL_DIR` | Διαδρομή μοντέλων που βλέπει ο API server. |
| `CSI_DB_PATH` | Διαδρομή SQLite αρχείου. Στο container είναι `/data/app.sqlite3`. |
| `CSI_COOKIE_SECURE` | Αν το browser cookie επιτρέπεται μόνο μέσω HTTPS. Να είναι `true` σε production. |
| `CSI_ALLOWED_HOSTS` | Επιτρεπόμενα Host headers. |
| `CSI_ALLOWED_ORIGINS` | Επιτρεπόμενα origins για WebSocket. |
| `CSI_ENABLE_API_DOCS` | Ενεργοποιεί ή όχι τα API docs· είναι κλειστό από προεπιλογή. |

Το `.env.example` είναι λίστα παραδείγματος, όχι αρχείο με έτοιμα production
secrets. Κάθε περιβάλλον πρέπει να έχει τα δικά του ισχυρά μυστικά.

## 12. Τοπικό και δημόσιο deployment δεν είναι το ίδιο

### Τοπική ανάπτυξη

- API: `./start_api.sh`
- Desktop: `./start_dashboard.sh`
- Website: `http://127.0.0.1:8000`
- cookie: για local HTTP μπορεί να είναι `CSI_COOKIE_SECURE=false`
- model files: τοπικά αρχεία από τα μοντέλα του project.

### Δημόσια ανάπτυξη

- API: Docker Compose, δεμένο μόνο σε loopback του server.
- Δημόσια είσοδος: Nginx με έγκυρο HTTPS certificate.
- Website: `https://το-domain-σου`
- Desktop: `CSI_SERVER_URL=https://το-domain-σου`
- ίδιο edge token στο desktop και στο API, κρατημένο μυστικό.
- `CSI_COOKIE_SECURE=true`.
- σωστά DNS, firewall, Host και Origin allowlists.
- μόνιμο Docker volume και backup της SQLite βάσης.

Δεν πρέπει να κάνεις port-forward τη θύρα 8000 δημόσια παρακάμπτοντας τον
Nginx. Το Compose είναι επίτηδες δεμένο στο loopback, ώστε η δημόσια είσοδος
να περνά από το TLS proxy.

## 13. Μοντέλα και Git LFS

Το API χρειάζεται τουλάχιστον:

- `csi_pipeline.joblib`,
- `label_encoder.joblib`,
- το επιλεγμένο classifier artifact.

Το `.dockerignore` δεν βάζει αυτά τα μεγάλα artifacts μέσα στο image. Το
Compose τα κάνει mount από τον host, read-only. Ο server ελέγχει αν τα αρχεία
είναι πραγματικά μοντέλα ή απλώς Git LFS pointer text και δίνει σαφές σφάλμα.

Πριν το deployment πρέπει να έχει γίνει Git LFS pull για τα απαιτούμενα model
directories. Αν το αντίστοιχο Motion artifact δεν έχει ληφθεί, η λειτουργία
Motion δεν μπορεί να ενεργοποιηθεί. Η αλλαγή mode/model δεν δημιουργεί νέο
εκπαιδευμένο μοντέλο· φορτώνει artifact που πρέπει ήδη να υπάρχει.

## 14. Τι γίνεται σε σφάλματα ή απώλεια δικτύου

- Ο desktop forwarder ξαναδοκιμάζει προσωρινά αποτυχημένα HTTP αιτήματα με
  αυξανόμενη αναμονή.
- Αν το API επιστρέψει σφάλμα που δείχνει άκυρο/απορριφθέν αίτημα ή token, ο
  forwarder σταματά τη συγκεκριμένη forwarding λειτουργία και εμφανίζει
  σφάλμα.
- Η ουρά των CSI frames είναι στη μνήμη και έχει όριο 5.000 frames.
- Αν ο desktop forwarder συνεχίσει να περιμένει τον server και η ουρά γεμίσει,
  το live dashboard καταγράφει ότι frames χάθηκαν από την προώθηση.
- Το command-line collector έχει επίσης bounded queue. Αν γεμίσει, το reader
  σταματά για να μην απορρίπτει σιωπηλά frames, αλλά η αποστολή ενός batch
  μπορεί να παραμένει σε retry όσο ο server δεν είναι προσβάσιμος.
- Δεν υπάρχει durable offline spool: restart/διακοπή ρεύματος στον edge
  υπολογιστή μπορεί να χάσει frames που δεν έχουν προωθηθεί.
- Ο API buffer είναι επίσης στη μνήμη. Με restart του API μηδενίζεται το
  τρέχον rolling inference context· η βάση και το ιστορικό παραμένουν.
- Το WebSocket επανασυνδέεται από τον browser, αλλά αυτό δεν επηρεάζει την
  αποστολή CSI από το desktop προς το API.

## 15. Όρια της τωρινής αρχιτεκτονικής

Αυτά είναι σημαντικά για να ξέρεις πού τελειώνει η σημερινή εγγύηση:

1. **Ένα API instance:** SQLite, buffer, inference model και WebSocket client
   list είναι δεμένα σε μία διεργασία/instance. Η κλιμάκωση σε πολλά replicas
   θέλει διαφορετικό σχεδιασμό, κοινό state και μηχανισμό broadcast.
2. **Ουρές μόνο στη RAM:** δεν εγγυάται διατήρηση όλων των CSI frames όταν ο
   υπολογιστής κλείσει ή το Internet μείνει κάτω για μεγάλο διάστημα.
3. **Δεν είναι πλήρες raw audit trail:** τα raw CSI frames δεν αποθηκεύονται
   και οι προβλέψεις/στατιστικές αποθηκεύονται δειγματοληπτικά όπως
   περιγράφεται στην ενότητα SQLite.
4. **Δεν υπάρχει MQTT:** δεν υπάρχει broker ή QoS/durable message queue.
5. **Δεν έχει αποδειχθεί failover:** το restart policy βοηθά να ξεκινήσει ξανά
   το container, αλλά δεν είναι high availability.
6. **Nginx configuration χρειάζεται ολοκλήρωση:** το example δεν εκδίδει
   certificates ούτε ρυθμίζει HTTP-to-HTTPS redirect.
7. **Κεντρικό edge token:** το ίδιο token δίνει πρόσβαση στα edge endpoints.
   Πολλές συσκευές/χρήστες θα απαιτούσαν ξεχωριστά credentials και δυνατότητα
   ανάκλησης ανά συσκευή.
8. **Πραγματική δοκιμή συσκευής:** τα unit tests δεν αντικαθιστούν δοκιμή με
   πραγματικό ESP32, μακρά λειτουργία, αποσύνδεση δικτύου και επανεκκίνηση.

Τα παραπάνω δεν σημαίνουν ότι το σύστημα δεν λειτουργεί. Περιγράφουν τα όρια
του τι εξασφαλίζει η σημερινή έκδοση και τι θα χρειαζόταν πριν από αυστηρές
απαιτήσεις παραγωγής.

## 16. Checklist πριν το θεωρήσεις σωστά deployed

### Πριν από το ξεκίνημα

- [ ] Έχουν κατέβει τα πραγματικά μοντέλα με Git LFS.
- [ ] Το `.env` υπάρχει μόνο στον server και δεν έχει μπει σε Git.
- [ ] Το admin password είναι ισχυρό και το edge token είναι τυχαίο, μοναδικό
  και τουλάχιστον 32 χαρακτήρες.
- [ ] Το domain δείχνει στη σωστή δημόσια IP.
- [ ] Τα `CSI_ALLOWED_HOSTS` και `CSI_ALLOWED_ORIGINS` είναι σωστά.
- [ ] Το TLS certificate είναι έγκυρο και ανανεώνεται.
- [ ] Το firewall δεν εκθέτει δημόσια την port 8000.
- [ ] Υπάρχει σχέδιο backup για το Docker volume/SQLite.

### Μετά το ξεκίνημα

- [ ] Το container είναι `healthy` στο `docker compose ps`.
- [ ] Το `/healthz` απαντά.
- [ ] Το `/readyz` απαντά και δείχνει το αναμενόμενο μοντέλο.
- [ ] Το website ανοίγει με `https://` χωρίς certificate warning.
- [ ] Ο admin μπορεί να κάνει login και logout.
- [ ] Το desktop συνδέεται στο σωστό HTTPS URL και στη σωστή serial port.
- [ ] Τα frame count/FPS αυξάνονται όταν εκπέμπει το ESP32.
- [ ] Εμφανίζονται προβλέψεις και στο desktop και στο website.
- [ ] Μία αλλαγή setting από το desktop φαίνεται στο website και παραμένει μετά
  από API restart.
- [ ] Έγινε δοκιμή backup και restore, όχι απλώς δημιουργία backup.

### Αν κάτι αποτύχει, πού κοιτάς πρώτα

1. **Δεν ανοίγει το site:** DNS, firewall 80/443, Nginx, TLS certificate.
2. **Nginx δείχνει 502:** αν το API container είναι έτοιμο και ακούει στο
   loopback port 8000.
3. **`/healthz` δουλεύει αλλά `/readyz` όχι:** logs API, μοντέλα, LFS αρχεία,
   `.env`.
4. **Το desktop δεν στέλνει frames:** serial port/permissions, `CSI_SERVER_URL`,
   token, TLS και API logs.
5. **Frames αυξάνονται αλλά δεν βγαίνει πρόβλεψη:** επαρκές warm-up/window,
   subcarrier count, συμβατότητα μοντέλου και inference logs.
6. **Το website δεν ενημερώνεται live:** browser login/session, WebSocket
   upgrade headers στο Nginx, browser console και επιτρεπόμενα origins.

Για τοπικό API/authentication/ingest test suite, από τον φάκελο
`Python_Scripts_for_csi_recv`:

```bash
.venv/bin/python -m pytest -q web_app/tests
```

Τα tests ελέγχουν την εφαρμογική συμπεριφορά με test client. Δεν ελέγχουν
DNS, TLS certificates, πραγματικό Nginx, serial hardware ή Internet
αποσυνδέσεις.

## 17. Τι να θυμάσαι

- Το **ESP32 μετρά**, το **desktop συλλέγει/προωθεί**, το **API προβλέπει και
  αποθηκεύει**, και το **website παρουσιάζει**.
- Με ενεργό API, το desktop δεν κάνει δεύτερο inference.
- HTTP/HTTPS μεταφέρει batches και αιτήματα ελέγχου· WebSocket μεταφέρει live
  ενημερώσεις στον browser.
- Το Docker τρέχει το API απομονωμένα· το volume κρατά τη βάση.
- Το Nginx είναι η δημόσια είσοδος και τερματίζει το HTTPS· δεν είναι το API
  και δεν κάνει inference.
- Για πραγματική χρήση μέσω Internet χρειάζονται DNS, TLS, firewall, μυστικά,
  persistence και επιβεβαίωση ολόκληρης της διαδρομής με πραγματική συσκευή.
- Το σημερινό σύστημα είναι σχεδιασμένο για ένα API instance και δεν παρέχει
  αδιάλειπτη αποθήκευση raw CSI κατά τη διάρκεια μεγάλης αποσύνδεσης.
