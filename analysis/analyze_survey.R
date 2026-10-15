library(tidyverse)
library(fixest)

# SETTINGS
data_directory <- "analysis/survey/data"
render_table_pdfs <- TRUE
seed <- 20260906
source("common.R") # MIN_PARTICIPANT_MINUTES is at the top of common.R.
results_directory <- "analysis/survey/results"
plot_directory <- "plots/survey"
for (directory in c(results_directory, plot_directory, file.path(plot_directory, "tables"))) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}

# SURVEY DATA: PARTICIPANT FILTER

survey_source <- jsonlite::fromJSON(file.path(data_directory, "embedding_manifest.json"))$input
participant_filter <- read_csv(file.path(survey_source, "sessions.csv"), show_col_types = FALSE) %>%
  transmute(
    prolific_pid,
    session_id,
    duration_minutes = as.numeric(difftime(completed_at, created_at, units = "mins")),
    minimum_minutes = MIN_PARTICIPANT_MINUTES,
    included = !is.na(completed_at) & !is.na(duration_minutes) & duration_minutes >= MIN_PARTICIPANT_MINUTES
  )

included_sessions <- participant_filter %>% filter(included) %>% pull(session_id)

participant_filter %>% write_csv(file.path(results_directory, "participant_filter.csv"))

Sys.setenv(XDG_CACHE_HOME = tempdir())
set.seed(seed)

# SURVEY DATA: PARTICIPANT MEANS
rounds <- read_csv(file.path(data_directory, "rounds.csv"), show_col_types = FALSE) %>%
  filter(included) %>%
  filter(session_id %in% included_sessions) %>%
  mutate(
    condition = factor(arm, levels = c("base", "trained"), labels = c("Base model", "Trained model")),
    description_word_change = output_words - request_words,
    dialogue_words = question_words + answer_words
  )

participants <- sort(unique(rounds$prolific_pid))
coverage <- rounds %>% count(prolific_pid, condition, name = "rounds")

# SURVEY OUTCOMES: EMBEDDINGS

# Keep all 15 process outcomes: they form the reported BH correction family.
process_labels <- c(
  n_questions_answered = "Questions answered",
  question_words = "Total question words",
  answer_words = "Total answer words",
  mean_question_words = "Words per question",
  mean_answer_words = "Words per answer",
  description_word_change = "Δ description words",
  dialogue_words = "Total dialogue words",
  round_seconds = "Total round time (s)",
  request_seconds = "Initial request time (s)",
  interaction_seconds = "Interaction time (s)",
  render_seconds = "Final generation time (s)",
  mean_answer_seconds = "Time per answer (s)",
  agent_reasoning_tokens = "Reasoning tokens",
  agent_output_tokens = "Visible output tokens",
  agent_total_tokens = "Total sampled tokens"
)
semantic_columns <- paste0(names(embedding_labels), "_cosine_gain")
outcomes <- rounds %>%
  select(
    prolific_pid,
    round_id,
    item_id,
    source_id,
    round_index,
    condition,
    all_of(c(semantic_columns, names(process_labels)))
  ) %>%
  pivot_longer(all_of(c(semantic_columns, names(process_labels))), names_to = "metric", values_to = "value") %>%
  filter(is.finite(value))
participant_means <- outcomes %>%
  group_by(prolific_pid, condition, metric) %>%
  summarize(value = mean(value), rounds = n(), .groups = "drop")
# SURVEY DATA: COVERAGE CHECK
coverage_summary <- rounds %>%
  group_by(condition) %>%
  summarize(participants = n_distinct(prolific_pid), rounds = n(), items = n_distinct(item_id), .groups = "drop") %>%
  left_join(
    coverage %>% group_by(condition) %>% summarize(min_rounds = min(rounds), max_rounds = max(rounds), .groups = "drop"),
    by = "condition"
  )
coverage_summary %>%
  write_csv(file.path(results_directory, "coverage_summary.csv"))

sprintf(
  "%s included participants; %s excluded sessions; %s rounds; %s items; %s items observed in both conditions.",
  length(participants),
  sum(!participant_filter$included),
  nrow(rounds),
  n_distinct(rounds$item_id),
  rounds %>% count(item_id, condition) %>% count(item_id) %>% filter(n == 2) %>% nrow()
) %>% writeLines(file.path(results_directory, "coverage_summary.txt"))

outcomes %>%
  write_csv(file.path(results_directory, "round_outcomes.csv"))

participant_means %>%
  write_csv(file.path(results_directory, "participant_means.csv"))


# SURVEY REGRESSIONS: IMAGE FIXED EFFECTS
regression_plan <- tibble(
  metric = c(
    paste0(names(embedding_labels), "_cosine_gain"),
    "n_questions_answered",
    "question_words",
    "round_seconds",
    "agent_total_tokens"
  ),
  table = rep(c("embedding_regressions", "process_regressions"), each = 4),
  label = c(
    "Text--text",
    "Image--image",
    "Text--image",
    "Image--text",
    "Questions",
    "Question words",
    "Round seconds",
    "Sampled tokens"
  )
)
regression_data <- outcomes %>%
  mutate(trained = as.numeric(condition == "Trained model")) %>%
  left_join(rounds %>% select(round_id, question_count = n_questions_answered), by = "round_id")
process_metrics <- regression_plan %>% filter(table == "process_regressions") %>% pull(metric)
survey_models <- map(
  process_metrics,
  ~feols(
    if (.x == "n_questions_answered") value ~ trained + factor(source_id) else value ~ trained + question_count + factor(source_id),
    fixef.rm = "none",
    data = filter(regression_data, metric == .x),
    cluster = ~prolific_pid,
    ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
  )
) %>% set_names(process_metrics)
regression_results <- map2_dfr(
  survey_models,
  names(survey_models),
  function(model, metric) {
    interval <- confint(model)["trained", ]
    tibble(
      metric = metric,
      estimate = coef(model)[["trained"]],
      se = se(model)[["trained"]],
      low = interval[[1]],
      high = interval[[2]],
      p_value = pvalue(model)[["trained"]],
      rounds = nobs(model),
      clusters = length(participants)
    )
  }
) %>%
  left_join(regression_plan, by = "metric")

regression_results %>% filter(!is.na(table)) %>%
  write_csv(file.path(results_directory, "regression_coefficients.csv"))

regression_plan %>%
  write_csv(file.path(results_directory, "regression_specifications.csv"))

# SURVEY TESTS: POST-HOC CONTRASTS

primary_metrics <- paste0(names(embedding_labels), "_cosine_gain")
plot_contrasts <- tibble()
condition_means <- tibble()
for (metric in unique(outcomes$metric)) {
  model_data <- regression_data %>% filter(metric == .env$metric)
  model_formula <- if (metric == "n_questions_answered") {
    value ~ trained + factor(source_id)
  } else {
    value ~ trained + question_count + factor(source_id)
  }
  model <- feols(
    model_formula,
    data = model_data,
    fixef.rm = "none",
    cluster = ~prolific_pid,
    ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
  )
  # Compare conditions at the pooled mean question count with equal image weights.
  means <- model %>%
    emmeans::emmeans(
      ~trained,
      data = model_data,
      at = list(trained = c(0, 1)),
      weights = "equal",
      vcov. = vcov(model),
      df = n_distinct(model_data$prolific_pid) - 1
    )
  mean_table <- means %>%
    confint() %>%
    as.data.frame() %>%
    transmute(
      metric = .env$metric,
      condition = factor(trained, levels = c(0, 1), labels = c("Base model", "Trained model")),
      estimate = emmean, se = SE, df, low = lower.CL, high = upper.CL
    ) %>%
    left_join(model_data %>% group_by(condition) %>% summarize(participants = n_distinct(prolific_pid), rounds = n(), .groups = "drop"), by = "condition")
  condition_means <- bind_rows(condition_means, mean_table)
  contrast <- means %>%
    emmeans::contrast(method = "trt.vs.ctrl", ref = 1, adjust = "none") %>%
    summary(infer = c(TRUE, TRUE)) %>%
    as.data.frame() %>%
    transmute(metric = .env$metric, estimate, se = SE, df, low = lower.CL, high = upper.CL, p_value = p.value)
  plot_contrasts <- bind_rows(plot_contrasts, contrast)
}
plot_contrasts <- plot_contrasts %>%
  mutate(family = case_when(
    metric %in% primary_metrics ~ "Primary quality outcomes",
    TRUE ~ "Process outcomes"
  )) %>%
  group_by(family) %>%
  mutate(
    bh_p_value = p.adjust(p_value, "BH"),
    stars = case_when(
      bh_p_value < 0.0001 ~ "****",
      bh_p_value < 0.001 ~ "***",
      bh_p_value < 0.01 ~ "**",
      bh_p_value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  ungroup()

plot_contrasts %>%
  write_csv(file.path(results_directory, "plot_contrasts.csv"))

condition_means %>% write_csv(file.path(results_directory, "condition_means.csv"))

# SURVEY PLOTS: STYLE
condition_colors <- c("Base model" = "#6B6875", "Trained model" = "#317873")
theme_set(
  theme_classic(base_size = 11, base_family = "Linux Libertine O") +
    theme(
      text = element_text(size = 11, color = "#29272B"),
      axis.title = element_text(size = 11),
      legend.text = element_text(size = 11),
      axis.line = element_line(linewidth = 0.3),
      axis.line.x.bottom = element_line(linewidth = 0.3),
      axis.line.y.left = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = 9),
      axis.text = element_text(size = 11, color = "#29272B"),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.spacing = grid::unit(8, "pt"),
      plot.margin = margin(4, 5, 4, 4)
    )
)

# SURVEY PLOTS: COSINE GAINS

plot_metrics <- paste0(names(embedding_labels), "_cosine_gain")
equation_labels <- unname(embedding_equations)
cosine_gains_plot <- mean_plot(plot_metrics, unname(embedding_labels), "Δ cosine similarity", equations = equation_labels)

cosine_gains_plot %>%
  ggsave(
    filename = file.path(plot_directory, "cosine_gains.pdf"),
    plot = .,
    width = figure_width,
    height = 2.9,
    device = grDevices::quartz,
    type = "pdf"
  )

# SURVEY PLOTS: INTERACTION BURDEN

plot_metrics <- c(
  "n_questions_answered",
  "question_words",
  "answer_words",
  "mean_question_words",
  "mean_answer_words",
  "dialogue_words",
  "round_seconds",
  "mean_answer_seconds"
)
interaction_burden_plot <- mean_plot(plot_metrics, process_labels[plot_metrics], "Adjusted mean", wrap = TRUE)

interaction_burden_plot %>%
  ggsave(
    filename = file.path(plot_directory, "survey_interaction_burden.pdf"),
    plot = .,
    width = figure_width,
    height = 4.2,
    device = grDevices::quartz,
    type = "pdf"
  )

# SURVEY PLOTS: TOKEN USAGE

plot_metrics <- c("agent_reasoning_tokens", "agent_output_tokens", "agent_total_tokens")
token_usage_plot <- mean_plot(plot_metrics, process_labels[plot_metrics], "Adjusted mean tokens")

token_usage_plot %>%
  ggsave(
    filename = file.path(plot_directory, "token_usage.pdf"),
    plot = .,
    width = figure_width,
    height = 2.6,
    device = grDevices::quartz,
    type = "pdf"
  )

# SURVEY PLOTS: TRAINING AT MATCHED QUESTION COUNTS

shared_counts <- rounds %>%
  distinct(n_questions_answered, condition) %>%
  count(n_questions_answered) %>%
  filter(n == 2) %>%
  pull(n_questions_answered)
burden_data <- regression_data %>% filter(metric %in% primary_metrics)
burden_support <- burden_data %>%
  group_by(metric, question_count, trained) %>%
  summarize(rounds = n(), participants = n_distinct(prolific_pid), items = n_distinct(item_id), .groups = "drop")

# Fix spline knots from the observed data before constructing the emmeans grid.
burden_knots <- quantile(burden_data$question_count, c(1 / 3, 2 / 3))
burden_boundaries <- range(burden_data$question_count)
burden_contrasts <- tibble()
for (metric in primary_metrics) {
  model_data <- burden_data %>% filter(metric == .env$metric)
  model <- feols(
    value ~ trained * splines::ns(question_count, knots = burden_knots, Boundary.knots = burden_boundaries) + factor(source_id),
    data = model_data,
    fixef.rm = "none",
    cluster = ~prolific_pid,
    ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
  )
  contrasts <- model %>%
    emmeans::emmeans(
      ~trained | question_count,
      data = model_data,
      at = list(trained = c(0, 1), question_count = shared_counts),
      weights = "equal",
      vcov. = vcov(model),
      df = n_distinct(model_data$prolific_pid) - 1
    ) %>%
    emmeans::contrast(method = "trt.vs.ctrl", ref = 1, adjust = "none") %>%
    summary(infer = c(TRUE, TRUE)) %>%
    as.data.frame() %>%
    transmute(metric = .env$metric, question_count, estimate, se = SE, df, low = lower.CL, high = upper.CL, p_value = p.value)
  burden_contrasts <- bind_rows(burden_contrasts, contrasts)
}
burden_contrasts <- burden_contrasts %>%
  mutate(
    bh_p_value = p.adjust(p_value, "BH"),
    stars = case_when(
      bh_p_value < 0.0001 ~ "****",
      bh_p_value < 0.001 ~ "***",
      bh_p_value < 0.01 ~ "**",
      bh_p_value < 0.05 ~ "*",
      TRUE ~ ""
    )
  )
burden_gains <- burden_contrasts %>%
  mutate(label = factor(metric, levels = primary_metrics, labels = unname(embedding_labels)))

burden_support %>% write_csv(file.path(results_directory, "quality_by_question_count_support.csv"))

burden_gains %>% write_csv(file.path(results_directory, "training_by_question_count.csv"))

burden_contrasts %>% write_csv(file.path(results_directory, "training_by_question_count_contrasts.csv"))

burden_star_labels <- burden_gains %>%
  filter(stars != "") %>%
  mutate(star_vjust = if_else(metric == "image_text_cosine_gain" & question_count == 4, -1.8, -0.25))

training_by_question_count_plot <- ggplot(burden_gains, aes(question_count, estimate)) +
  geom_hline(yintercept = 0, color = "#96939D", linewidth = 0.3, linetype = "dashed") +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.25, linewidth = 0.35, color = condition_colors[["Trained model"]]) +
  geom_point(size = 1.5, color = condition_colors[["Trained model"]]) +
  geom_text(data = burden_star_labels, aes(y = high, label = stars, vjust = star_vjust), size = 11 / .pt, family = "Linux Libertine O") +
  ggh4x::facet_grid2(cols = vars(label), scales = "free_y", independent = "y", axes = "all", labeller = label_wrap_gen(width = 20)) +
  scale_x_continuous(breaks = c(0, 5, 10, 15)) +
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.15))) +
  labs(x = "Questions answered", y = "Trained − base cosine gain")

training_by_question_count_plot %>%
  ggsave(
    filename = file.path(plot_directory, "training_by_question_count.pdf"),
    plot = .,
    width = figure_width,
    height = 2.9,
    device = grDevices::quartz,
    type = "pdf"
  )

plot_specs <- tibble(
  name = c("cosine_gains", "survey_interaction_burden", "token_usage", "training_by_question_count"),
  width = rep(figure_width, 4),
  height = c(2.9, 4.2, 2.6, 2.9)
)
plot_specs %>%
  write_csv(file.path(results_directory, "plot_manifest.csv"))

cat(sprintf("Completed survey analysis: %s participants, %s completed rounds, %s figures.\n", length(participants), nrow(rounds), nrow(plot_specs)))

for (table in "process_regressions") {
  indices <- which(regression_plan$table == table)
  metrics <- process_metrics
  display_models <- map(metrics, ~summary(survey_models[[.x]]))
  output <- etable(
    display_models,
    tex = TRUE,
    dict = c(trained = "Trained model", question_count = "Questions answered", source_id = "Source image", prolific_pid = "Participant ID"),
    depvar = FALSE,
    headers = list("Outcome" = regression_plan$label[indices]),
    digits = 3,
    digits.stats = 3,
    se.below = TRUE,
    float = FALSE,
    style.tex = style.tex(main = "base", model.title = "Model:"),
    fitstat = ~n + r2,
    drop = "factor\\(source_id\\)",
    extralines = list("Source image indicators" = rep("Yes", length(display_models))),
    signif.code = c("****" = 0.0001, "***" = 0.001, "**" = 0.01, "*" = 0.05)
  )
  output <- output[seq(
    which(str_detect(output, fixed("\\begin{tabular}"))),
    which(str_detect(output, fixed("\\end{tabular}")))
  )]
  writeLines(
    c("{\\setlength{\\tabcolsep}{2.5pt}", output, "}"),
    file.path(plot_directory, "tables", paste0(table, ".tex"))
  )
}

# SURVEY TABLES: EMBEDDING SPECIFICATIONS

embedding_specifications <- tibble(
  specification = c("Training", "Training + source FE", "Adjusted", "Adjusted + source FE"),
  formula = c("value ~ trained", "value ~ trained + factor(source_id)", "value ~ trained + question_count", "value ~ trained + question_count + factor(source_id)")
)
embedding_plan <- regression_plan %>%
  filter(table == "embedding_regressions") %>%
  crossing(embedding_specifications) %>%
  mutate(
    specification = factor(specification, levels = embedding_specifications$specification),
    metric = factor(metric, levels = paste0(names(embedding_labels), "_cosine_gain"))
  ) %>%
  arrange(metric, specification)
embedding_models <- list()
embedding_coefficients <- tibble()
for (i in seq_len(nrow(embedding_plan))) {
  model <- feols(
    as.formula(embedding_plan$formula[i]),
    data = regression_data %>% filter(metric == embedding_plan$metric[i]),
    fixef.rm = "none",
    cluster = ~prolific_pid,
    ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
  )
  embedding_models[[i]] <- model
  coefficients <- coeftable(model) %>%
    as.data.frame() %>%
    rownames_to_column("term") %>%
    set_names(c("term", "estimate", "se", "statistic", "p_value")) %>%
    mutate(
      metric = embedding_plan$metric[i],
      specification = embedding_plan$specification[i],
      low = confint(model)[, 1],
      high = confint(model)[, 2],
      rounds = nobs(model),
      clusters = length(participants)
    )
  embedding_coefficients <- bind_rows(embedding_coefficients, coefficients)
}
embedding_plan %>%
  write_csv(file.path(results_directory, "embedding_regression_specifications.csv"))

embedding_coefficients %>%
  write_csv(file.path(results_directory, "embedding_regression_coefficients.csv"))

for (metric in unique(embedding_plan$metric)) {
  indices <- which(embedding_plan$metric == metric)
  display_models <- embedding_models[indices]
  output <- etable(
    display_models,
    tex = TRUE,
    dict = c(trained = "Trained model", question_count = "Questions answered", source_id = "Source image", prolific_pid = "Participant ID"),
    depvar = FALSE,
    headers = list("Outcome" = rep(embedding_plan$label[indices[1]], 4), "Specification" = c("Training", "Training + FE", "Adjusted", "Adjusted + FE")),
    digits = 3,
    digits.stats = 3,
    se.below = TRUE,
    float = FALSE,
    style.tex = style.tex(main = "base", model.title = "Model:"),
    fitstat = ~n + r2,
    drop = "factor\\(source_id\\)",
    extralines = list("Source image indicators" = c("No", "Yes", "No", "Yes")),
    notes = character(),
    signif.code = c("****" = 0.0001, "***" = 0.001, "**" = 0.01, "*" = 0.05)
  )
  output <- output[seq(
    which(str_detect(output, fixed("\\begin{tabular}"))),
    which(str_detect(output, fixed("\\end{tabular}")))
  )]
  writeLines(
    output,
    file.path(plot_directory, "tables", paste0("quality_", metric, "_regressions.tex"))
  )
}

# SURVEY EXPORTS: TABLE PDFS

if (render_table_pdfs) {
  table_sources <- file.path(
    plot_directory,
    "tables",
    c(paste0("quality_", primary_metrics, "_regressions.tex"), "process_regressions.tex")
  )
  render_latex_tables(table_sources, survey = TRUE)
}
