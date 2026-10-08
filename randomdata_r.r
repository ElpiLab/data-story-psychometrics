# ==============================================================================
# Wissenschaftliche Lohn- und Persönlichkeitsanalyse (TREE2 Cohort Data) - v7
# Inklusive automatischer Datensatz-Bereinigung & Export als CSV / RDS / SPSS
# ==============================================================================
 
# 1. Benötigte Pakete laden
library(tidyverse)    # Datenbereinigung und Visualisierung
library(haven)        # Import & Export von SPSS-Dateien (.sav)
library(lmtest)       # Hypothesentests für Regressionen
library(sandwich)     # Cluster-robuste Standardfehler
library(modelsummary) # Tabellarische Aufbereitung der Modelle
 
# ------------------------------------------------------------------------------
# 2. Datenimport & Merging
# ------------------------------------------------------------------------------
# Persönlichkeitsdaten (CSV)
data_big5 <- read_csv("C:/Users/vinze/OneDrive/Dokumente/Fhnw/Fächer/Spezielles/Datasets/TREE Study/TREE2_Data_BIG5.csv", col_types = cols()) %>%
  mutate(resp_id = as.character(resp_id))
 
# Episodendaten (SPSS .sav) 
data_episodes <- read_sav("C:/Users/vinze/OneDrive/Dokumente/Fhnw/Fächer/Spezielles/Datasets/TREE Study/1 DATASETS/TREE2_Data_Episodes_v3.sav") %>% 
  zap_labels() %>% 
  mutate(resp_id = as.character(resp_id))
 
# Filtern auf Erwerbsepisoden (epi_type == 1) und Zusammenführung über resp_id
data_merged <- data_episodes %>%
  filter(epi_type == 1) %>% # 1 = Employment / Job
  inner_join(data_big5, by = "resp_id")
 
cat("Anzahl verknüpfter Erwerbsepisoden:", nrow(data_merged), "\n")
 
# ------------------------------------------------------------------------------
# 3. Bereinigung von Fehlwerten (TREE2 Missing Codes)
# ------------------------------------------------------------------------------
missing_codes <- c(-999, -998, -997, -996, -990, -989, -988, -960, -959, -901, -8001)
 
df_clean <- data_merged %>%
  mutate(across(where(is.numeric), ~ ifelse(. %in% missing_codes | . < 0, NA, .))) %>%
  filter(!is.na(salaam_init) & salaam_init > 0) %>%
  filter(!is.na(jobh) & jobh > 0)
 
cat("Anzahl Erwerbsepisoden mit gültigem Lohn & Arbeitszeit:", nrow(df_clean), "\n")
 
# ------------------------------------------------------------------------------
# 4. Variablenkonstruktion & Transformation
# ------------------------------------------------------------------------------
df_analysestichprobe <- df_clean %>%
  mutate(
    # A) Stundenlohn berechnen (1 = Stundenlohn, 2 = Monatslohn, 7 = Jahreslohn)
    stundenlohn = case_when(
      salacat_init == 1 ~ salaam_init,
      salacat_init == 2 ~ salaam_init / (jobh * 4.333),
      salacat_init == 7 ~ salaam_init / (jobh * 52),
      TRUE ~ NA_real_
    )
  ) %>%
  # Plausibilitätsfilter für Stundenlohn (z. B. 5 bis 500 CHF/h)
  filter(!is.na(stundenlohn) & stundenlohn >= 5 & stundenlohn <= 500) %>%
  mutate(
    log_stundenlohn = log(stundenlohn),
    # B) Z-Standardisierung der Big Five
    z_offenheit          = as.numeric(scale(t0big5_o_comp)),
    z_gewissenhaftigkeit = as.numeric(scale(t0big5_c_comp)),
    z_extraversion       = as.numeric(scale(t0big5_e_comp)),
    z_vertraeglichkeit   = as.numeric(scale(t0big5_a_comp)),
    z_neurotizismus      = as.numeric(scale(t0big5_n_comp)),
    # C) Kontrollvariablen
    gross_salary_dummy    = ifelse(sala4_init == 1, 1, 0),
    fuehrungsposition_num = ifelse(jpos3_init == 1, 1, 0),
    berufsstatus_isei     = as.numeric(job_isei08),
    wochenstunden         = as.numeric(jobh)
  )
 
# Optionale Faktorvariablen aufbereiten
if("educ_isced11" %in% names(df_analysestichprobe)) {
  df_analysestichprobe$bildungsniveau <- droplevels(as.factor(df_analysestichprobe$educ_isced11))
}
if("cobur_headcount" %in% names(df_analysestichprobe)) {
  df_analysestichprobe$betriebsgroesse <- droplevels(as.factor(df_analysestichprobe$cobur_headcount))
}
 
cat("Gültige Beobachtungen für Lohnanalyse:", nrow(df_analysestichprobe), "\n")
 
# ------------------------------------------------------------------------------
# 5. Hilfsfunktion zur sicheren Modellschätzung (Sicheres Handling von Missing Values)
# ------------------------------------------------------------------------------
fit_safe_lm <- function(base_formula, data) {
  all_vars <- all.vars(base_formula)
  existing_vars <- intersect(all_vars, names(data))
  sub_data <- data %>% select(all_of(existing_vars)) %>% drop_na()
  if(nrow(sub_data) == 0) {
    stop("Fehler: Keine vollständigen Fälle für diese Formulierung vorhanden.")
  }
  dep_var <- all_vars[1]
  indep_vars <- setdiff(existing_vars, dep_var)
  valid_indep <- c()
  for(v in indep_vars) {
    vals <- sub_data[[v]]
    if(is.factor(vals) || is.character(vals)) {
      if(length(unique(vals)) >= 2) valid_indep <- c(valid_indep, v)
    } else {
      if(sd(vals, na.rm = TRUE) > 0) valid_indep <- c(valid_indep, v)
    }
  }
  final_formula <- as.formula(paste(dep_var, "~", paste(valid_indep, collapse = " + ")))
  req_cols <- intersect(c(dep_var, valid_indep, "resp_id"), names(data))
  fit_data <- data %>% drop_na(all_of(req_cols))
  model <- lm(final_formula, data = fit_data)
  return(list(model = model, data = fit_data, formula = final_formula, n = nrow(fit_data)))
}
 
# ------------------------------------------------------------------------------
# 6. Modellschätzungen (Hierarchischer Aufbau)
# ------------------------------------------------------------------------------
 
# Modell 1: Unadjustierter Roheffekt der Big Five
f1 <- log_stundenlohn ~ z_offenheit + z_gewissenhaftigkeit + z_extraversion + 
                        z_vertraeglichkeit + z_neurotizismus
res1 <- fit_safe_lm(f1, df_analysestichprobe)
m1 <- res1$model
 
# Modell 2: Mincer-Basismodell (+ Brutto-Dummy & Arbeitszeit)
f2 <- log_stundenlohn ~ z_offenheit + z_gewissenhaftigkeit + z_extraversion + 
                        z_vertraeglichkeit + z_neurotizismus +
                        gross_salary_dummy + wochenstunden
res2 <- fit_safe_lm(f2, df_analysestichprobe)
m2 <- res2$model
 
# Modell 3: Vollständiges Arbeitsmarktmodell (+ ISEI Berufsstatus & Führungsposition)
f3 <- log_stundenlohn ~ z_offenheit + z_gewissenhaftigkeit + z_extraversion + 
                        z_vertraeglichkeit + z_neurotizismus +
                        gross_salary_dummy + wochenstunden +
                        berufsstatus_isei + fuehrungsposition_num
res3 <- fit_safe_lm(f3, df_analysestichprobe)
m3 <- res3$model
 
# Cluster-robuste Standardfehler berechnen (geclustert nach Person: resp_id)
vcov1 <- vcovCL(m1, cluster = ~resp_id, data = res1$data)
vcov2 <- vcovCL(m2, cluster = ~resp_id, data = res2$data)
vcov3 <- vcovCL(m3, cluster = ~resp_id, data = res3$data)
 
# ------------------------------------------------------------------------------
# 7. Regressionsergebnisse ausgeben
# ------------------------------------------------------------------------------
cat("\n=== REGRESSIONSERGEBNISSE (CLUSTER-ROBUSTE STANDARDFEHLER) ===\n")
msummary(
  list("Modell 1 (Unadjustiert)" = m1, 
       "Modell 2 (+ Arbeitsmarkt-Basics)" = m2, 
       "Modell 3 (Vollmodell)" = m3),
  vcov = list(vcov1, vcov2, vcov3),
  stars = TRUE,
  gof_omit = "IC|Log"
)
 
# ------------------------------------------------------------------------------
# 8. EXPORT DES BEREINIGTEN ANALYSIS-DATASETS
# ------------------------------------------------------------------------------
# Erstellung des finalen, bereinigten Datensatzes für das Vollmodell
clean_analysis_dataset <- res3$data %>%
  select(
    # Identifikator
    resp_id,
    # Zielvariablen (Lohn)
    stundenlohn,
    log_stundenlohn,
    salaam_init,
    salacat_init,
    # Big Five Persönlichkeitsfaktoren (Z-standardisiert & Rohwerte)
    z_offenheit, z_gewissenhaftigkeit, z_extraversion, z_vertraeglichkeit, z_neurotizismus,
    t0big5_o_comp, t0big5_c_comp, t0big5_e_comp, t0big5_a_comp, t0big5_n_comp,
    # Kontrollvariablen
    gross_salary_dummy,
    wochenstunden,
    berufsstatus_isei,
    fuehrungsposition_num,
    any_of(c("bildungsniveau", "betriebsgroesse"))
  )
 
# Export-Pfad im selben Verzeichnis ablegen
export_dir <- "C:/Users/vinze/OneDrive/Dokumente/Fhnw/Fächer/Spezielles/Datasets/TREE Study/"
 
# Datensatz im CSV-, RDS- und SPSS (.sav) Format speichern
write_csv(clean_analysis_dataset, paste0(export_dir, "TREE2_Clean_Analysis_Dataset.csv"))
write_rds(clean_analysis_dataset, paste0(export_dir, "TREE2_Clean_Analysis_Dataset.rds"))
write_sav(clean_analysis_dataset, paste0(export_dir, "TREE2_Clean_Analysis_Dataset.sav"))
 
cat("\n=== EXPORT ERFOLGREICH ===")
cat("\nDer bereinigte Datensatz wurde unter folgendem Pfad gespeichert:\n")
cat(export_dir, "\n")
cat("- CSV:  TREE2_Clean_Analysis_Dataset.csv\n")
cat("- RDS:  TREE2_Clean_Analysis_Dataset.rds (Empfohlen für R)\n")
cat("- SPSS: TREE2_Clean_Analysis_Dataset.sav\n")