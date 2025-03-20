# Required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

plot_trajectory <- function(microbe_name, subject, pred_df, labels_df) {
  # Filter the data for the specified subject
  pred_filtered <- pred_df %>% filter(subject_id == subject)
  labels_filtered <- labels_df %>% filter(subject_id == subject)
  
  # Extract the relevant microbial data
  pred_microbe <- pred_filtered %>% select(time, all_of(microbe_name))
  labels_microbe <- labels_filtered %>% select(time, all_of(microbe_name))
  
  # Merge predicted and real data by time
  merged_data <- merge(pred_microbe, labels_microbe, by = "time", suffixes = c("_pred", "_real"))
  
  # Fit linear model to calculate R²
  model <- lm(get(paste0(microbe_name, "_real")) ~ get(paste0(microbe_name, "_pred")), data = merged_data)
  r_squared <- summary(model)$r.squared
  
  # Calculate confidence intervals for the linear model
  conf_interval <- predict(model, interval = "confidence")
  
  # Add the confidence intervals to the merged data
  merged_data <- cbind(merged_data, conf_interval)
  
 
  p <- ggplot(merged_data, aes(x = time)) +
    geom_ribbon(aes(ymin = lwr, ymax = upr), fill = "grey80", alpha = 0.5) +
    geom_smooth(aes(y = get(paste0(microbe_name, "_pred")), color = "Predicted"), 
                method = "loess", size = 1, se = FALSE) +
    geom_smooth(aes(y = get(paste0(microbe_name, "_real")), color = "Actual"), 
                method = "loess", size = 1, se = FALSE) +
    labs(title = paste(microbe_name, "(", subject, ")"),
         x = "Time",
         y = "Abundance") +
    annotate("text", x = max(merged_data$time) * 0.8, 
             y = max(merged_data$upr) * 0.8, 
             label = paste0("R² = ", round(r_squared, 2)), size = 5) +
    scale_color_manual(values = c("Predicted" = "#7998AD", "Actual" = "#F07673")) +
    scale_y_continuous(limits = c(0, NA)) +  # Ensure y-axis starts at 0
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 13),
      legend.title = element_blank(),
      legend.text = element_text(size = 12),
      panel.border = element_rect(color = "black", fill = NA, size = 1) # Add border
    ) +
    theme(aspect.ratio = 0.8) # Set aspect ratio to make the plot shorter
  
  return(p)
}

# Define combinations
combinations <- data.frame(
  microbe_name = c(
    "g__Acinetobacter",
    "g__Bacillus",
    "g__Blvii28",
    "g__Chryseobacterium",
    "g__Exiguobacterium",
    "g__Flavobacterium",
    "g__Flectobacillus",
    "g__Herbaspirillum",
    "g__Lactococcus",
    "g__Leuconostoc",
    "g__Lysinibacillus",
    "g__Ochrobactrum",
    "g__Pedobacter",
    "g__Pseudomonas",
    "g__Ruminococcus",
    "g__Sphingobacterium",
    "g__Staphylococcus",
    "g__Stenotrophomonas",
    "g__Wautersiella"
  ),
  subject = rep("7F", 19)  
)


# Generate plots
plots <- lapply(1:nrow(combinations), function(i) {
  plot_trajectory(
    microbe_name = combinations$microbe_name[i],
    subject = combinations$subject[i],
    pred_df = pred,
    labels_df = true
  )
})


pdf("trajectory_plots_all.pdf", width = 15, height = 12) 
grid.arrange(grobs = plots, ncol = 4, nrow = 5)  
dev.off()









