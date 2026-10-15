# SURVEY SETTINGS

MIN_PARTICIPANT_MINUTES <- 25

# LABELS

embedding_labels <- c(
  text = "Text to reference text",
  image = "Image to reference image",
  text_image = "Text to reference image",
  image_text = "Image to reference text"
)

model_labels <- c(
  "base-model" = "Base model",
  "patient-trained" = "Patient-trained",
  "impatient-trained" = "Impatient-trained"
)
policy_labels <- c(
  "no-asking" = "No questions",
  "fixed-asking" = "Fixed asking",
  "open-asking" = "Open asking"
)
user_labels <- c("patient-user" = "Patient user", "impatient-user" = "Impatient user")
model_colors <- c(
  "Base model" = "#6B6875",
  "Patient-trained" = "#317873",
  "Impatient-trained" = "#AB6776"
)
policy_colors <- c("No questions" = "#96939D", "Fixed asking" = "#317873", "Open asking" = "#AB6776")
policy_shapes <- c("No questions" = 15, "Fixed asking" = 16, "Open asking" = 17)
policy_lines <- c("No questions" = "dotted", "Fixed asking" = "solid", "Open asking" = "longdash")
metric_labels <- c(
  "utility_importance" = "Importance-weighted recovery",
  "utility_equal" = "Equal-weighted recovery",
  "no_recovery" = "Episodes recovering no hidden details",
  "full_recovery" = "Episodes recovering every hidden detail",
  "answered_questions" = "Questions answered",
  "questions_per_detail" = "Answered questions per missing detail",
  "zero_questions" = "Episodes with no answered questions",
  "question_words" = "Total question words",
  "output_words" = "Final description words",
  "answer_words" = "Total answer words",
  "mean_question_words" = "Words per question",
  "mean_answer_words" = "Words per answer",
  "dialogue_words" = "Dialogue words",
  "cap_fraction" = "Answered share of cap",
  "cap_reached" = "Episodes reaching cap",
  "text_image_progress" = unname(embedding_labels["text_image"]),
  "image_text_progress" = unname(embedding_labels["image_text"]),
  "text_hidden_gain" = "Text to redacted details",
  "text_visible_gain" = "Text to visible details",
  "image_hidden_gain" = "Image to redacted details",
  "image_visible_gain" = "Image to visible details",
  "text_progress" = unname(embedding_labels["text"]),
  "image_progress" = unname(embedding_labels["image"]),
  "text_detail_selectivity" = "Text: redacted minus visible",
  "image_detail_selectivity" = "Image: redacted minus visible",
  "text_similarity" = "Final text / full-description cosine",
  "image_similarity" = "Generated / reference-image cosine"
)
metric_index <- enframe(metric_labels, "metric", "metric_label") %>%
  mutate(
    family = case_when(
      str_starts(metric, "utility") ~ "Recovery",
      metric %in% c("no_recovery", "full_recovery") ~ "Reliability",
      str_starts(metric, "text_|image_") ~ "Semantic",
      TRUE ~ "Interaction burden"
    )
  )

# EMBEDDING NOTATION

embedding_equations <- c(
  text_progress = "plain(sim)(T[plain(fin)], T[plain(ref)]) - plain(sim)(T[plain(init)], T[plain(ref)])",
  image_progress = "plain(sim)(I[plain(fin)], I[plain(ref)]) - plain(sim)(I[plain(init)], I[plain(ref)])",
  text_image_progress = "plain(sim)(T[plain(fin)], I[plain(ref)]) - plain(sim)(T[plain(init)], I[plain(ref)])",
  image_text_progress = "plain(sim)(I[plain(fin)], T[plain(ref)]) - plain(sim)(I[plain(init)], T[plain(ref)])"
)

# FONT
quartzFonts("Linux Libertine O" = quartzFont(c("LinLibertineO", "LinLibertineOB", "LinLibertineOI", "LinLibertineOBI")))

# FIGURE DIMENSIONS
figure_width <- 8.5 - 108 / 72.27
column_width <- (figure_width - 24 / 72.27) / 2
font_size <- 11
embedding_metrics <- c(
  "text_progress",
  "image_progress",
  "text_image_progress",
  "image_text_progress",
  "text_hidden_gain",
  "image_hidden_gain",
  "text_visible_gain",
  "image_visible_gain",
  "text_detail_selectivity",
  "image_detail_selectivity"
)
primary_metrics <- c("utility_equal", "text_progress", "image_progress")
metric_labels[primary_metrics] <- c("Recovery", "Text to full reference text", "Image to reference image")
metric_index <- metric_index %>% mutate(metric_label = unname(metric_labels[metric]))
for (directory in c(
  "analysis/data",
  "analysis/results",
  "plots",
  "plots/tables"
)) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}
theme_set(
  theme_classic(base_size = font_size, base_family = "Linux Libertine O") +
    theme(
      text = element_text(size = font_size, color = "#29272B"),
      axis.text = element_text(size = font_size, color = "#29272B"),
      axis.title = element_text(size = font_size),
      axis.line = element_line(linewidth = 0.3),
      axis.line.x.bottom = element_line(linewidth = 0.3),
      axis.line.y.left = element_line(linewidth = 0.3),
      axis.ticks = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = font_size),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = font_size),
      legend.key.height = grid::unit(11, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      plot.margin = margin(4, 5, 4, 4),
      panel.spacing = grid::unit(6, "pt")
    )
)

significance_stars <- function(p) {
  case_when(is.na(p) ~ "", p < 0.0001 ~ "****", p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "")
}

adjust_tests <- function(data, groups) {
  data %>%
    group_by(across(all_of(groups))) %>%
    mutate(bh_p_value = p.adjust(p_value, "BH"), stars = significance_stars(bh_p_value)) %>%
    ungroup()
}

# MODEL ESTIMATED MARGINAL MEANS
model_condition_emmeans <- function(data, metric, adjust_count = TRUE) {
  data <- data %>% mutate(condition = factor(condition, levels = sort(unique(condition))))
  count_outcomes <- c("answered_questions", "questions_per_detail", "cap_fraction", "cap_reached", "zero_questions")
  model_formula <- if (adjust_count && !metric %in% count_outcomes) {
    value ~ condition + answered_questions + factor(source_id)
  } else {
    value ~ condition + factor(source_id)
  }
  model <- feols(
    model_formula,
    data = data,
    fixef.rm = "none",
    cluster = ~source_id,
    ssc = ssc(K.adj = TRUE, K.fixef = "full", G.adj = TRUE, G.df = "min", t.df = "min")
  )
  emmeans::emmeans(
    model,
    ~condition,
    data = data,
    weights = "equal",
    vcov. = vcov(model),
    df = n_distinct(data$source_id) - 1
  )
}

# MODEL CONDITION CONTRASTS
model_control_contrasts <- function(data, comparisons, metric, adjust_count = TRUE) {
  condition_levels <- sort(unique(data$condition))
  comparisons <- comparisons %>% filter(left %in% condition_levels, right %in% condition_levels)
  counts <- comparisons %>%
    select(contrast, left, right) %>%
    mutate(n = map2_int(left, right, ~length(intersect(data$source_id[data$condition == .x], data$source_id[data$condition == .y])))) %>%
    select(contrast, n)
  if (n_distinct(data$source_id) < 2) {
    return(counts %>% mutate(estimate = NA_real_, se = NA_real_, df = NA_real_, low = NA_real_, high = NA_real_, p_value = NA_real_))
  }
  means <- model_condition_emmeans(data, metric, adjust_count = adjust_count)
  coefficients <- map2(
    comparisons$left,
    comparisons$right,
    ~as.numeric(condition_levels == .x) - as.numeric(condition_levels == .y)
  ) %>%
    set_names(comparisons$contrast)
  emmeans::contrast(means, method = coefficients, adjust = "none") %>%
    summary(infer = c(TRUE, TRUE)) %>%
    as.data.frame() %>%
    transmute(contrast, estimate, se = SE, df, low = lower.CL, high = upper.CL, p_value = p.value) %>%
    left_join(counts, by = "contrast")
}

# DESCRIPTIVE SOURCE MEANS: INTERCEPT-MODEL EMMS
source_summary <- function(data) {
  source_values <- data %>%
    group_by(source_id) %>%
    summarize(value = mean(value), .groups = "drop")
  if (nrow(source_values) == 1) {
    return(tibble(n = 1L, estimate = source_values$value, low = NA_real_, high = NA_real_))
  }
  model <- lm(value ~ 1, data = source_values)
  model %>%
    emmeans::emmeans(~1) %>%
    confint() %>%
    as.data.frame() %>%
    transmute(n = nrow(source_values), estimate = emmean, low = lower.CL, high = upper.CL)
}

save_figure <- function(plot, name, height = 4, width = figure_width, equation = NULL) {
  if (inherits(plot, "patchwork")) {
    plot <- (plot & labs(title = NULL, subtitle = NULL, caption = NULL)) +
      plot_annotation(title = NULL, subtitle = NULL, caption = NULL)
  } else {
    plot <- plot + labs(title = NULL, subtitle = NULL, caption = NULL)
  }
  if (!is.null(equation)) {
    plot <- plot +
      labs(subtitle = parse(text = equation)[[1]]) +
      theme(plot.subtitle = element_text(size = 9, hjust = 0.5))
  }
  ggsave(
    file.path("plots", paste0(name, ".pdf")),
    plot,
    width = width,
    height = height,
    device = grDevices::quartz,
    type = "pdf"
  )
}

# TABLE FRAGMENTS
tidy_model <- function(model, outcome, specification) {
  coefficients <- as.data.frame(coeftable(model))
  intervals <- confint(model)
  tibble(
    specification = specification,
    metric = outcome,
    term = rownames(coefficients),
    estimate = coefficients[[1]],
    se = coefficients[[2]],
    low = intervals[, 1],
    high = intervals[, 2],
    p_value = coefficients[[4]],
    observations = nobs(model)
  )
}

# MODEL PLOT HELPER

overview_plot <- function(metrics, axis_label, show_training = TRUE, equations = NULL) {
  metric_labeller <- label_wrap_gen(16)
  if (!is.null(equations)) {
    math_labels <- paste0("atop(plain(\"", metric_labels[metrics], "\"), scriptstyle(", equations, "))") %>%
      set_names(metric_labels[metrics])
    metric_labeller <- as_labeller(math_labels, default = label_parsed)
  }
  plot <- plot_means %>% filter(metric %in% metrics) %>%
    ggplot(aes(policy, estimate, color = model, group = model)) +
    geom_errorbar(aes(ymin = low, ymax = high), position = position_dodge(0.55), width = 0.10, linewidth = 0.5) +
    geom_point(position = position_dodge(0.55), size = 1.5) +
    facet_grid(
      metric_label ~ evaluation,
      scales = "free",
      space = "free_x",
      axes = "all",
      axis.labels = "margins",
      labeller = labeller(metric_label = metric_labeller, evaluation = label_wrap_gen(16))
    ) +
    scale_color_manual(values = model_colors, name = "Model checkpoint") +
    scale_x_discrete(labels = c("No questions" = "None", "Fixed asking" = "Fixed", "Open asking" = "Open")) +
    scale_y_continuous(n.breaks = 4, expand = expansion(mult = c(0.06, 0.10))) +
    guides(color = guide_legend(nrow = 1)) +
    labs(x = "Asking policy", y = axis_label) +
    theme(legend.title = element_text(size = font_size), strip.text.y = element_text(angle = 0))
  if (!is.null(equations)) plot <- plot + theme(strip.text.y = element_text(size = 9, angle = 0))
  if (length(metrics) == 1) plot <- plot + theme(strip.text.y = element_blank())
  if (show_training) {
    brackets <- overview_brackets %>% filter(metric %in% metrics)
    plot <- plot +
      geom_blank(
        data = brackets,
        aes(x = control_x, y = y + if_else(metric == "utility_equal", 0, 0.18 * span)),
        inherit.aes = FALSE
      ) +
      geom_segment(
        data = brackets,
        aes(x = control_x, xend = treatment_x, y = y, yend = y),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_segment(
        data = brackets,
        aes(x = control_x, xend = control_x, y = y, yend = y - tip),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_segment(
        data = brackets,
        aes(x = treatment_x, xend = treatment_x, y = y, yend = y - tip),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_text(
        data = brackets,
        aes(x = (control_x + treatment_x) / 2, y = y, label = stars),
        inherit.aes = FALSE,
        family = "Linux Libertine O",
        size = font_size / .pt,
        vjust = 0.5
      )
  }
  plot
}

# SURVEY PLOT HELPER

mean_plot <- function(metrics, labels, axis_label, wrap = FALSE, equations = NULL) {
  estimates <- condition_means %>%
    filter(metric %in% metrics) %>%
    mutate(label = factor(metric, levels = metrics, labels = labels))
  if (!is.null(equations)) {
    estimates <- estimates %>% mutate(equation = factor(equations[match(metric, metrics)], levels = equations))
  }
  brackets <- estimates %>%
    group_by(metric, label) %>%
    summarize(top = max(high), span = max(high) - min(low), .groups = "drop") %>%
    left_join(plot_contrasts %>% select(metric, bh_p_value, stars), by = "metric") %>%
    filter(bh_p_value < 0.05) %>%
    mutate(span = pmax(span, abs(top) * 0.05, 0.01), y = top + 0.12 * span, tip = 0.04 * span)
  if (all(str_detect(metrics, "_cosine_gain$"))) {
    # Keep significance labels clear of the zero reference line.
    brackets <- brackets %>% mutate(y = if_else(abs(y) < 0.15 * span, 0.15 * span, y))
  }
  if (!is.null(equations)) {
    brackets <- brackets %>% mutate(equation = factor(equations[match(metric, metrics)], levels = equations))
  }
  facets <- if (wrap) {
    facet_wrap(vars(label), ncol = 4, scales = "free_y", axes = "all", labeller = label_wrap_gen(width = 16))
  } else if (!is.null(equations)) {
    ggh4x::facet_grid2(
      cols = vars(label, equation),
      scales = "free_y",
      independent = "y",
      axes = "all",
      labeller = labeller(label = label_value, equation = label_parsed),
      strip = ggh4x::strip_themed(
        text_x = ggh4x::elem_list_text(size = c(8.5, 7), family = "Linux Libertine O", margin = list(margin(2, 0, 2, 0))),
        by_layer_x = TRUE
      )
    )
  } else {
    ggh4x::facet_grid2(
      cols = vars(label),
      scales = "free_y",
      independent = "y",
      axes = "all",
      labeller = label_wrap_gen(width = 16)
    )
  }
  plot <- ggplot(estimates, aes(condition, estimate)) +
    geom_errorbar(aes(ymin = low, ymax = high, color = condition), width = 0.08, linewidth = 0.45) +
    geom_point(aes(color = condition), size = 2.1) +
    facets +
    scale_color_manual(values = condition_colors) +
    labs(x = NULL, y = axis_label, color = NULL) +
    theme(axis.text.x = element_blank(), legend.position = "bottom")
  if (nrow(brackets)) {
    plot <- plot +
      geom_segment(data = brackets, aes(x = 1, xend = 2, y = y, yend = y), inherit.aes = FALSE, linewidth = 0.3) +
      geom_segment(data = brackets, aes(x = 1, xend = 1, y = y, yend = y - tip), inherit.aes = FALSE, linewidth = 0.3) +
      geom_segment(data = brackets, aes(x = 2, xend = 2, y = y, yend = y - tip), inherit.aes = FALSE, linewidth = 0.3) +
      geom_text(
        data = brackets,
        aes(x = 1.5, y = y, label = stars),
        inherit.aes = FALSE,
        family = "Linux Libertine O",
        size = 11 / .pt,
        vjust = 0.5
      ) +
      geom_blank(data = brackets, aes(x = 1.5, y = y + 0.18 * span), inherit.aes = FALSE)
  }
  if (all(str_detect(metrics, "_cosine_gain$"))) {
    plot <- plot + geom_hline(yintercept = 0, color = "#C8C5CC", linewidth = 0.3)
  }
  plot
}

# TABLE PDF RENDERING

render_latex_tables <- function(sources, survey = FALSE) {
  build_directory <- tempfile("paper-tables-")
  dir.create(build_directory)
  progress <- txtProgressBar(min = 0, max = length(sources), style = 3)

  for (i in seq_along(sources)) {
    source <- sources[i]
    name <- tools::file_path_sans_ext(basename(source))
    document <- file.path(build_directory, paste0(name, ".tex"))
    input <- paste0("\\input{", normalizePath(source), "}")

    preamble <- c(
      "\\documentclass[11pt,border=8pt]{standalone}",
      "\\usepackage{libertine,booktabs,graphicx}"
    )
    body <- input

    c(preamble, "\\begin{document}", body, "\\end{document}") %>%
      writeLines(document)

    system2(
      "pdflatex",
      c("-interaction=nonstopmode", "-halt-on-error", paste0("-output-directory=", shQuote(build_directory)), shQuote(document)),
      stdout = file.path(build_directory, paste0(name, ".log.txt"))
    )

    file.copy(
      file.path(build_directory, paste0(name, ".pdf")),
      sub("[.]tex$", ".pdf", source),
      overwrite = TRUE
    )
    setTxtProgressBar(progress, i)
  }
  close(progress)
}
