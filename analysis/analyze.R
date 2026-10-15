library(tidyverse)
library(fixest)
library(patchwork)

# SETTINGS

use_embeddings <- TRUE
seed <- 20260905
render_table_pdfs <- TRUE
set.seed(seed)
Sys.setenv(XDG_CACHE_HOME = tempdir())
source("common.R")
source("embedding_helpers.R")

# MODEL DATA: CONDITIONS

raw <- read_csv("clip_dataset/dataset.csv", show_col_types = FALSE)
metadata <- jsonlite::fromJSON("clip_dataset/metadata.json", simplifyVector = FALSE)
run_index <- tibble(condition = unlist(metadata$agents)) %>%
  separate_wider_delim(
    condition,
    delim = "_",
    names = c("model_code", "policy_code", "user_code"),
    cols_remove = FALSE
  ) %>%
  mutate(
    model = factor(model_labels[model_code], levels = unname(model_labels)),
    policy = factor(policy_labels[policy_code], levels = unname(policy_labels)),
    user = factor(user_labels[user_code], levels = unname(user_labels)),
    shared_baseline = policy_code == "no-asking",
    panel = factor(
      if_else(shared_baseline, "Shared no-question baseline", as.character(user)),
      levels = c("Shared no-question baseline", "Patient user", "Impatient user")
    ),
    condition_label = paste(
      model,
      policy,
      if_else(shared_baseline, "Shared baseline", as.character(user)),
      sep = " | "
    )
  ) %>%
  arrange(model, policy, user)

instances <- raw %>%
  select(
    dataset,
    instance_id,
    source_id,
    split,
    n_details,
    n_redacted,
    ground_truth_text,
    request_text
  ) %>%
  mutate(
    episode_id = paste(dataset, instance_id, sep = "::"),
    k_group = cut(
      n_redacted,
      breaks = c(0, 5, 10, 15),
      labels = c("1-5 missing details", "6-10 missing details", "11-15 missing details")
    ),
    request_words = str_count(request_text, boundary("word")),
    full_description_words = str_count(ground_truth_text, boundary("word"))
  )
methods <- map_dfr(
  run_index$condition,
  function(condition) {
    raw %>%
      transmute(
        dataset,
        instance_id,
        condition = .env$condition,
        output_text = .data[[paste0(condition, "_output_text")]],
        output_image_path = .data[[paste0(condition, "_output_image_path")]],
        question_cap = .data[[paste0(condition, "_question_cap")]],
        answered_questions = .data[[paste0(condition, "_n_questions")]],
        utility_equal = .data[[paste0(condition, "_u_rec_equal")]],
        question_texts = map(.data[[paste0(condition, "_question_texts")]], jsonlite::fromJSON),
        answer_texts = map(.data[[paste0(condition, "_answer_texts")]], jsonlite::fromJSON)
      )
  },
  .progress = "Reading conditions"
) %>%
  left_join(instances, by = c("dataset", "instance_id")) %>%
  left_join(run_index, by = "condition") %>%
  mutate(
    output_words = str_count(output_text, boundary("word"))
  )

# MODEL DATA: DIALOGUE TURNS
turns <- read_csv("clip_dataset/turns.csv", show_col_types = FALSE) %>%
  rename(condition = agent) %>%
  left_join(
    instances %>% select(dataset, instance_id, source_id, episode_id, n_redacted),
    by = c("dataset", "instance_id")
  ) %>%
  left_join(run_index, by = "condition") %>%
  arrange(episode_id, condition, turn) %>%
  group_by(episode_id, condition) %>%
  mutate(
    question_words = str_count(question, boundary("word")),
    answer_words = str_count(answer, boundary("word"))
  ) %>%
  ungroup()
turn_totals <- turns %>%
  group_by(episode_id, condition) %>%
  summarize(
    mean_question_words = mean(question_words),
    mean_answer_words = mean(answer_words),
    question_words = sum(question_words),
    answer_words = sum(answer_words),
    .groups = "drop"
  )
methods <- methods %>%
  left_join(turn_totals, by = c("episode_id", "condition")) %>%
  mutate(
    across(
      c(question_words, answer_words),
      ~replace_na(.x, 0)
    ),
    mean_question_words = if_else(answered_questions > 0, mean_question_words, NA_real_),
    mean_answer_words = if_else(answered_questions > 0, mean_answer_words, NA_real_),
    dialogue_words = question_words + answer_words
  )
run_index %>%
  write_csv("analysis/data/conditions.csv")

instances %>%
  write_csv("analysis/data/instances.csv")

# EMBEDDINGS: OUTCOME SCORES

if (use_embeddings) {

  embedded <- arrow::read_parquet("gemini_2_embeddings.parquet")

  semantic_index <- crossing(row = seq_len(nrow(embedded)), condition = run_index$condition)
  semantic_methods <- pmap_dfr(
    semantic_index,
    function(row, condition) {
      text_output <- embedded[[paste0(condition, "_output_text_embedding")]][[row]]
      image_output <- embedded[[paste0(condition, "_output_image_embedding")]][[row]]
      tibble(
        episode_id = paste(embedded$dataset[row], embedded$instance_id[row], sep = "::"),
        condition = condition,
        text_progress = cosine_gain(
          embedded$request_text_embedding[[row]],
          text_output,
          embedded$ground_truth_text_embedding[[row]]
        ),
        image_progress = cosine_gain(
          embedded$request_image_embedding[[row]],
          image_output,
          embedded$ground_truth_enhanced_image_embedding[[row]]
        ),
        text_image_progress = cosine_gain(
          embedded$request_text_embedding[[row]],
          text_output,
          embedded$ground_truth_enhanced_image_embedding[[row]]
        ),
        image_text_progress = cosine_gain(
          embedded$request_image_embedding[[row]],
          image_output,
          embedded$ground_truth_text_embedding[[row]]
        )
      )
    },
    .progress = "Computing target semantics"
  )
  methods <- methods %>%
    left_join(semantic_methods, by = c("episode_id", "condition"))

}

# MODEL ESTIMATES: OUTCOMES

source_ids <- sort(unique(methods$source_id))
# Retain complete contrast families for the paper's BH-adjusted tests.
analysis_metrics <- c(
  "utility_equal", "text_progress", "image_progress", "text_image_progress", "image_text_progress",
  "answered_questions", "question_words", "answer_words", "dialogue_words", "mean_question_words", "mean_answer_words"
)
outcome_long <- methods %>%
  select(source_id, condition, all_of(analysis_metrics)) %>%
  pivot_longer(all_of(analysis_metrics), names_to = "metric", values_to = "value") %>%
  filter(is.finite(value))
# MODEL TESTS: REGRESSION-ADJUSTED CONTRASTS
contrast_index <- crossing(left = run_index$condition, right = run_index$condition) %>%
  left_join(run_index, by = c("left" = "condition")) %>%
  left_join(run_index, by = c("right" = "condition"), suffix = c("_left", "_right")) %>%
  mutate(
    set = case_when(
      model_code_left != "base-model" & model_code_right == "base-model" & policy_code_left == policy_code_right & user_code_left == user_code_right ~ "Training",
      model_code_left == model_code_right & policy_code_left == "open-asking" & policy_code_right == "fixed-asking" & user_code_left == user_code_right ~ "Asking policy",
      model_code_left == model_code_right & policy_code_left == policy_code_right & user_code_left == "impatient-user" & user_code_right == "patient-user" ~ "User patience",
      model_code_left == model_code_right & policy_code_left != "no-asking" & policy_code_right == "no-asking" ~ "Asking"
    )
  ) %>%
  filter(!is.na(set)) %>%
  mutate(contrast = paste(left, right, sep = " vs "))
paired_values <- contrast_index %>%
  select(contrast, left, right) %>%
  inner_join(outcome_long, by = c("left" = "condition"), relationship = "many-to-many") %>%
  rename(treatment = value) %>%
  inner_join(outcome_long, by = c("right" = "condition", "source_id", "metric")) %>%
  mutate(value = treatment - value)
contrast_data <- outcome_long %>%
  left_join(
    methods %>% select(source_id, condition, answered_questions),
    by = c("source_id", "condition")
  )
condition_summary <- contrast_data %>%
  group_by(metric) %>%
  group_modify(function(data, key) {
    model_condition_emmeans(data, key$metric) %>%
      confint() %>%
      as.data.frame() %>%
      transmute(condition = as.character(condition), estimate = emmean, se = SE, df, low = lower.CL, high = upper.CL) %>%
      left_join(data %>% group_by(condition) %>% summarize(n = n_distinct(source_id), .groups = "drop"), by = "condition")
  }) %>%
  ungroup() %>%
  left_join(run_index, by = "condition")

contrast_tests <- contrast_data %>%
  group_by(metric) %>%
  group_modify(
    function(data, key) {
      model_control_contrasts(data, contrast_index, key$metric)
    }
  ) %>%
  ungroup() %>%
  adjust_tests("metric") %>%
  left_join(contrast_index, by = "contrast")
condition_summary %>%
  write_csv("analysis/results/condition_means.csv")

contrast_tests %>%
  write_csv("analysis/results/control_contrasts.csv")

# MODEL REGRESSIONS: FACTORIAL DESIGN

regression_data <- methods %>%
  filter(!shared_baseline) %>%
  mutate(
    checkpoint = factor(model_code, levels = c("base-model", "patient-trained", "impatient-trained")),
    open = as.integer(policy_code == "open-asking"),
    impatient = as.integer(user_code == "impatient-user"),
    any_answer = as.integer(answered_questions > 0),
    question_length = replace_na(mean_question_words, 0),
    answer_length = replace_na(mean_answer_words, 0),
    output_hundreds = output_words / 100
  )

# MODEL REGRESSIONS: SPECIFICATIONS
specifications <- c(
  "Additive" = "checkpoint + open + impatient",
  "Two-way" = "checkpoint * open + checkpoint * impatient + open * impatient",
  "Three-way" = "checkpoint * open * impatient",
  "+ Count" = "checkpoint * open * impatient + any_answer + answered_questions",
  "+ Q. length" = "checkpoint * open * impatient + any_answer + answered_questions + question_length",
  "+ A./output" = "checkpoint * open * impatient + any_answer + answered_questions + question_length + answer_length + output_hundreds"
)
regression_plan <- bind_rows(
  crossing(table = "recovery_regressions", metric = "utility_equal", specification = names(specifications)),
  crossing(
    table = "semantic_regressions",
    metric = c("text_progress", "image_progress"),
    specification = c("Additive", "Three-way", "+ A./output")
  ),
  crossing(
    table = "crossmodal_regressions",
    metric = c("text_image_progress", "image_text_progress"),
    specification = c("Additive", "Three-way", "+ A./output")
  ),
  crossing(table = "burden_regressions", metric = "answered_questions", specification = "Additive"),
  crossing(
    table = "burden_regressions",
    metric = c("question_words", "answer_words"),
    specification = c("Additive", "Three-way")
  ),
  crossing(table = "burden_regressions", metric = "output_words", specification = "Three-way")
) %>%
  filter(metric %in% names(regression_data)) %>%
  mutate(
    specification = factor(specification, levels = names(specifications)),
    metric = factor(
      metric,
      levels = c("utility_equal", "text_progress", "image_progress", "text_image_progress", "image_text_progress", "answered_questions", "question_words", "answer_words", "output_words")
    )
  ) %>%
  arrange(table, metric, specification) %>%
  mutate(model_id = paste(metric, specification, sep = " / "))
regression_models <- map2(
  regression_plan$metric,
  regression_plan$specification,
  function(metric, specification) {
    feols(
      as.formula(paste(metric, "~", specifications[[specification]], "+ factor(source_id)")),
      data = regression_data,
      fixef.rm = "none",
      cluster = ~source_id,
      ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
    )
  },
  .progress = "Fitting factorial specification sets"
) %>%
  set_names(regression_plan$model_id)

regression_results <- pmap_dfr(
  select(regression_plan, model_id, metric, specification, table),
  function(model_id, metric, specification, table) {
    tidy_model(regression_models[[model_id]], as.character(metric), as.character(specification)) %>%
      mutate(model_id = model_id, table = table)
  }
) %>%
  mutate(stars = significance_stars(p_value))
regression_results %>%
  write_csv("analysis/results/regression_coefficients.csv")

regression_plan %>%
  write_csv("analysis/results/regression_specifications.csv")

# MODEL TABLES: REGRESSION SETS
regression_labels <- c(
  "checkpointpatient-trained" = "Patient-trained",
  "checkpointimpatient-trained" = "Impatient-trained",
  "open" = "Open",
  "impatient" = "Impatient",
  "any_answer" = "Any answer",
  "answered_questions" = "Questions",
  "question_length" = "Words/question",
  "answer_length" = "Words/answer",
  "output_hundreds" = "Output words / 100",
  "source_id" = "Source image",
  "utility_equal" = "Equal-weighted recovery",
  "text_progress" = "Text--text",
  "image_progress" = "Image--image",
  "text_image_progress" = "Text--image",
  "image_text_progress" = "Image--text",
  "question_words" = "Question words",
  "answer_words" = "Answer words",
  "output_words" = "Output words"
)
save_regression_set <- function(table) {
  columns <- regression_plan %>% filter(.data$table == .env$table)
  models <- regression_models[columns$model_id]
  latex <- etable(
    models,
    tex = TRUE,
    dict = regression_labels,
    headers = list("Specification" = as.character(columns$specification)),
    digits = 3,
    digits.stats = 3,
    se.below = TRUE,
    fitstat = ~n + r2,
    drop = "factor\\(source_id\\)",
    extralines = list("Source image indicators" = rep("Yes", length(models))),
    signif.code = c("****" = .0001, "***" = .001, "**" = .01, "*" = .05),
    float = FALSE,
    style.tex = style.tex(main = "base", model.title = "Model:")
  )
  latex <- latex[seq(
    which(str_detect(latex, fixed("\\begin{tabular}"))),
    which(str_detect(latex, fixed("\\end{tabular}")))
  )]
  latex <- str_replace_all(latex, fixed("source\\_id"), "source")
  writeLines(
    c("{\\setlength{\\tabcolsep}{2.5pt}", latex, "}"),
    file.path("plots/tables", paste0(table, ".tex"))
  )
  regression_results %>%
    filter(.data$table == .env$table) %>%
    select(metric, specification, term, estimate, se, low, high, p_value, stars, observations) %>%
    write_csv(file.path("analysis/results", paste0(table, ".csv")))
}
for (table_name in unique(regression_plan$table)) {
  save_regression_set(table_name)
}

# MODEL PLOTS: PERFORMANCE WITHOUT QUESTION-COUNT ADJUSTMENT
performance_metrics <- c("utility_equal", "text_progress", "image_progress", "text_image_progress", "image_text_progress")
performance_means <- contrast_data %>%
  filter(metric %in% performance_metrics) %>%
  group_by(metric) %>%
  group_modify(function(data, key) {
    model_condition_emmeans(data, key$metric, adjust_count = FALSE) %>%
      confint() %>%
      as.data.frame() %>%
      transmute(condition = as.character(condition), estimate = emmean, se = SE, df, low = lower.CL, high = upper.CL)
  }) %>%
  ungroup()
performance_tests <- contrast_data %>%
  filter(metric %in% performance_metrics) %>%
  group_by(metric) %>%
  group_modify(function(data, key) {
    model_control_contrasts(data, contrast_index, key$metric, adjust_count = FALSE)
  }) %>%
  ungroup() %>%
  adjust_tests("metric") %>%
  left_join(contrast_index, by = "contrast")
performance_means %>% write_csv("analysis/results/performance_means.csv")
performance_tests %>% write_csv("analysis/results/performance_contrasts.csv")

# MODEL PLOTS: SCALES

plot_means <- condition_summary %>%
  mutate(
    evaluation = factor(
      if_else(shared_baseline, "No interaction", as.character(user)),
      levels = c("No interaction", unname(user_labels))
    ),
    metric_label = factor(unname(metric_labels[metric]), levels = unname(metric_labels[analysis_metrics])),
    across(c(estimate, low, high), ~if_else(metric %in% c("utility_equal"), 100 * .x, .x))
  )
plot_means <- plot_means %>%
  left_join(performance_means %>% select(metric, condition, estimate, low, high), by = c("metric", "condition"), suffix = c("", "_performance")) %>%
  mutate(
    estimate = coalesce(estimate_performance * if_else(metric == "utility_equal", 100, 1), estimate),
    low = coalesce(low_performance * if_else(metric == "utility_equal", 100, 1), low),
    high = coalesce(high_performance * if_else(metric == "utility_equal", 100, 1), high)
  ) %>%
  select(-ends_with("_performance"))
plot_means %>%
  write_csv("analysis/results/plotted_overview_means.csv")

# MODEL PLOTS: RECOVERY AND EMBEDDING SCORES
overview_groups <- plot_means %>%
  group_by(metric) %>%
  mutate(span = max(high) - min(low)) %>%
  group_by(metric, evaluation, metric_label, policy) %>%
  summarize(top = max(high), span = first(span), .groups = "drop") %>%
  group_by(metric, evaluation) %>%
  mutate(policy_x = row_number()) %>%
  ungroup()
overview_tests <- bind_rows(contrast_tests %>% filter(!metric %in% performance_metrics), performance_tests)
overview_brackets <- overview_tests %>%
  filter(set == "Training", bh_p_value < 0.05) %>%
  inner_join(
    plot_means %>% select(condition, evaluation, metric_label, policy, metric),
    by = c("left" = "condition", "metric")
  ) %>%
  left_join(overview_groups, by = c("metric", "evaluation", "metric_label", "policy")) %>%
  group_by(metric, evaluation, policy) %>%
  arrange(model_code_left != "patient-trained", .by_group = TRUE) %>%
  mutate(
    tier = row_number(),
    control_x = policy_x - 0.55 / 3,
    treatment_x = policy_x + if_else(model_code_left == "patient-trained", 0, 0.55 / 3),
    gap = if_else(metric == "utility_equal", 4, 0.06 * span),
    step = if_else(metric == "utility_equal", 8, 0.18 * span),
    tip = if_else(metric == "utility_equal", 2, 0.025 * span),
    y = top + gap + (tier - 1) * step
  ) %>%
  ungroup()
overview_brackets %>%
  write_csv("analysis/results/plotted_overview_training_brackets.csv")

overview_brackets %>% filter(metric == "utility_equal") %>%
  write_csv("analysis/results/plotted_recovery_training_brackets.csv")

recovery_plot <- overview_plot("utility_equal", "Recovery (%)", show_training = TRUE) +
  scale_y_continuous(breaks = seq(0, 100, 25), limits = c(0, 100), expand = expansion(mult = c(0, 0.02)))
save_figure(recovery_plot, "recovery", height = 2.4)
if (use_embeddings) {
  overview_plot(
    c("text_progress", "image_progress", "text_image_progress", "image_text_progress"),
    "Δ cosine similarity",
    show_training = TRUE,
    equations = unname(embedding_equations)
  ) %>%
    save_figure("embedding_scores", height = 4.6)

}

# MODEL PLOTS: TASK DIFFICULTY
difficulty_summary <- methods %>%
  mutate(value = utility_equal) %>%
  group_by(condition, n_redacted) %>%
  group_modify(~source_summary(.x)) %>%
  ungroup() %>%
  left_join(run_index, by = "condition")
difficulty_summary %>%
  write_csv("analysis/results/recovery_by_difficulty.csv")

difficulty_plot <- difficulty_summary %>% filter(!shared_baseline) %>%
  ggplot(aes(n_redacted, 100 * estimate, color = model, group = model)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.5) +
  facet_grid(policy ~ user, axes = "all", axis.labels = "margins") +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(breaks = c(1, 5, 10, 15)) +
  labs(x = "Missing details", y = "Mean recovery (%)")
save_figure(difficulty_plot, "task_difficulty", height = 3.3)

# MODEL PLOTS: ANSWERED TURN COVERAGE
turn_coverage <- methods %>%
  filter(!shared_baseline) %>%
  select(source_id, condition, answered_questions) %>%
  crossing(turn = seq_len(max(turns$turn))) %>%
  mutate(value = as.numeric(answered_questions >= turn)) %>%
  group_by(condition, turn) %>%
  group_modify(~source_summary(.x)) %>%
  ungroup() %>%
  left_join(run_index, by = "condition")
turn_coverage %>%
  write_csv("analysis/results/answered_turn_coverage.csv")

coverage_plot <- ggplot(turn_coverage, aes(turn, estimate, color = model, linetype = policy)) +
  geom_step(linewidth = 0.6) +
  facet_grid(user ~ model, axes = "all", axis.labels = "margins") +
  scale_color_manual(values = model_colors, guide = "none") +
  scale_linetype_manual(values = policy_lines) +
  scale_y_continuous(labels = scales::label_percent()) +
  labs(x = "Answered turn", y = "Episodes reaching this turn")
save_figure(coverage_plot, "answered_turn_coverage", height = 3.3)

# MODEL PLOTS: QUESTION COUNTS
question_counts <- methods %>%
  filter(!shared_baseline) %>%
  group_by(condition, n_redacted) %>%
  summarize(estimate = mean(answered_questions), .groups = "drop") %>%
  left_join(run_index, by = "condition")
question_counts %>% write_csv("analysis/results/question_count_by_difficulty.csv")
question_count_plot <- ggplot(question_counts, aes(n_redacted, estimate, color = model, linetype = policy)) +
  geom_line(linewidth = 0.6) +
  facet_grid(user ~ model, axes = "all", axis.labels = "margins") +
  scale_color_manual(values = model_colors, guide = "none") +
  scale_linetype_manual(values = policy_lines) +
  scale_x_continuous(breaks = c(1, 5, 10, 15)) +
  labs(x = "Missing details", y = "Mean answered questions")
save_figure(question_count_plot, "question_count_by_difficulty", height = 4.5)

# MODEL PLOTS: TRAINING GAIN AND EFFORT
training_burden_values <- paired_values %>%
  filter(metric == "utility_equal") %>%
  inner_join(
    contrast_index %>% filter(set == "Training", !shared_baseline_left),
    by = "contrast",
    suffix = c("", "_index")
  ) %>%
  left_join(
    methods %>% select(source_id, condition, baseline_questions = answered_questions),
    by = c("source_id", "right" = "condition")
  )
training_changes <- training_burden_values %>%
  left_join(
    methods %>% select(source_id, condition, answered_questions, question_words, answer_words),
    by = c("source_id", "left" = "condition")
  ) %>%
  left_join(
    methods %>% select(source_id, condition, baseline_question_words = question_words, baseline_answer_words = answer_words),
    by = c("source_id", "right" = "condition")
  ) %>%
  mutate(
    question_change = answered_questions - baseline_questions,
    question_word_change = question_words - baseline_question_words,
    answer_word_change = answer_words - baseline_answer_words
  )
training_changes %>% select(
  source_id,
  contrast,
  model_left,
  policy_left,
  user_left,
  question_change,
  question_word_change,
  answer_word_change,
  recovery_change = value
) %>%
  write_csv("analysis/results/training_burden_changes.csv")

effort_labels <- c(
  "question_change" = "Change in answered questions",
  "question_word_change" = "Change in total question words",
  "answer_word_change" = "Change in total answer words"
)
training_effort_changes <- training_changes %>%
  filter(policy_left == "Open asking") %>%
  select(
    source_id,
    contrast,
    model_left,
    policy_left,
    user_left,
    recovery_change = value,
    all_of(names(effort_labels))
  ) %>%
  pivot_longer(all_of(names(effort_labels)), names_to = "effort", values_to = "effort_change") %>%
  mutate(
    model_left = factor(model_left, levels = unname(model_labels)),
    user_left = factor(user_left, levels = unname(user_labels))
  )
training_effort_changes %>%
  write_csv("analysis/results/training_effort_changes.csv")

effort_gain_limits <- 25 * c(
  floor(min(100 * training_effort_changes$recovery_change) / 25),
  ceiling(max(100 * training_effort_changes$recovery_change) / 25)
)
effort_plots <- map(
  names(effort_labels),
  function(effort) {
    plot <- training_effort_changes %>% filter(.data$effort == .env$effort) %>%
      ggplot(aes(effort_change, 100 * recovery_change, color = model_left)) +
      geom_hline(yintercept = 0, color = "#96939D", linewidth = 0.3) +
      geom_vline(xintercept = 0, color = "#96939D", linewidth = 0.3) +
      geom_point(alpha = 0.35, size = 1.6, shape = 1, stroke = 0.5) +
      geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.35) +
      facet_grid(. ~ user_left, axes = "all", axis.labels = "margins") +
      scale_color_manual(values = model_colors) +
      guides(color = guide_legend(override.aes = list(alpha = 1, shape = 16))) +
      scale_y_continuous(breaks = seq(-100, 100, 50)) +
      coord_cartesian(ylim = effort_gain_limits) +
      labs(x = effort_labels[[effort]], y = "Recovery gain (pp)")
    if (effort != "question_change") plot <- plot + theme(strip.text = element_blank())
    plot
  }
)
change_plot <- wrap_plots(effort_plots, ncol = 1, guides = "collect") & theme(legend.position = "bottom")
save_figure(change_plot, "training_gain_and_question_change", height = 5.2)

# MODEL EXPORTS
methods %>% select(-question_texts, -answer_texts) %>% write_csv("analysis/data/episodes.csv")
turns %>% write_csv("analysis/data/turns.csv")
writeLines(capture.output(sessionInfo()), "analysis/results/session_info.txt")

# MODEL TABLES: PDF PREVIEWS
if (render_table_pdfs) {
  table_sources <- file.path("plots/tables", paste0(unique(regression_plan$table), ".tex"))
  render_latex_tables(table_sources)
}
