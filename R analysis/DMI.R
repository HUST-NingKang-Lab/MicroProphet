library(vegan)
library(dplyr)
library(tidyr)

data <- read.csv("Low nutrient.csv")
data$time <- as.factor(data$time)

unique_genera <- colnames(data)[-(1:2)]

dmi_results <- data.frame()

for (genus in unique_genera) {
  
  genus_data <- data %>% select(subject_id, time, !!sym(genus))
  colnames(genus_data)[3] <- "value"
  
  intra_bc_list <- list()
  inter_bc_list <- list()
  
  unique_subjects <- unique(genus_data$subject_id)
  for (subject in unique_subjects) {
    subject_data <- genus_data %>% filter(subject_id == subject)
    if (nrow(subject_data) < 2) next
    
    subject_features <- as.matrix(subject_data[, "value", drop = FALSE])
    intra_bc <- vegdist(subject_features, method = "bray")
    intra_bc_list[[subject]] <- as.vector(intra_bc)
  }
  
  unique_times <- unique(genus_data$time)
  for (time_point in unique_times) {
    time_data <- genus_data %>% filter(time == time_point)
    if (nrow(time_data) < 2) next
    
    time_features <- as.matrix(time_data[, "value", drop = FALSE])
    inter_bc <- vegdist(time_features, method = "bray")
    inter_bc_list[[time_point]] <- as.vector(inter_bc)
  }
  
  intra_median <- median(unlist(intra_bc_list), na.rm = TRUE)
  inter_median <- median(unlist(inter_bc_list), na.rm = TRUE)
  
  dmi <- inter_median - intra_median
  
  n_iterations <- 1000
  bootstrap_dmi <- replicate(n_iterations, {
    resampled_data <- genus_data[sample(1:nrow(genus_data), replace = TRUE), ]
    
    resampled_intra_bc_list <- list()
    for (subject in unique(resampled_data$subject_id)) {
      subject_data <- resampled_data %>% filter(subject_id == subject)
      if (nrow(subject_data) < 2) next
      subject_features <- as.matrix(subject_data[, "value", drop = FALSE])
      resampled_intra_bc <- vegdist(subject_features, method = "bray")
      resampled_intra_bc_list[[subject]] <- as.vector(resampled_intra_bc)
    }
    
    resampled_inter_bc_list <- list()
    for (time_point in unique(resampled_data$time)) {
      time_data <- resampled_data %>% filter(time == time_point)
      if (nrow(time_data) < 2) next
      time_features <- as.matrix(time_data[, "value", drop = FALSE])
      resampled_inter_bc <- vegdist(time_features, method = "bray")
      resampled_inter_bc_list[[time_point]] <- as.vector(resampled_inter_bc)
    }
    
    resampled_intra_median <- median(unlist(resampled_intra_bc_list), na.rm = TRUE)
    resampled_inter_median <- median(unlist(resampled_inter_bc_list), na.rm = TRUE)
    
    resampled_dmi <- resampled_inter_median - resampled_intra_median
    return(resampled_dmi)
  })
  
  bootstrap_dmi <- bootstrap_dmi[!is.na(bootstrap_dmi)]
  
  dmi_sd <- sd(bootstrap_dmi, na.rm = TRUE)
  
  if (length(bootstrap_dmi) > 0 && !is.na(dmi_sd) && dmi_sd <= 15 * mean(bootstrap_dmi, na.rm = TRUE)) {
    dmi_results <- bind_rows(dmi_results, data.frame(
      genus = genus,
      intra_median = intra_median,
      inter_median = inter_median,
      DMI = dmi,
      DMI_SD = dmi_sd
    ))
  }
}

write.csv(dmi_results, "dmi_results.csv", row.names = FALSE)

library(ggplot2)
library(dplyr)
library(readr)

low_dmi <- read_csv("Low dmi results.csv")
high_dmi <- read_csv("High dmi results.csv")

low_dmi <- low_dmi %>% select(Genus = 1, DMI = 4) %>% mutate(Group = "Low nutrient")
high_dmi <- high_dmi %>% select(Genus = 1, DMI = 4) %>% mutate(Group = "High nutrient")

dmi_data <- bind_rows(low_dmi, high_dmi)

pdf("true_DMI_low_high.pdf", width = 5, height = 5) 
library(ggpubr)
library(ggplot2)
library(dplyr)
library(ggsignif)

quantiles <- dmi_data %>%
  group_by(Group) %>%
  summarise(
    median = median(DMI),
    q25 = quantile(DMI, 0.25),
    q75 = quantile(DMI, 0.75)
  )
map_signif_level<-function(p){
  if(p<0.001){
    p=formatC(p,format="e",digits=2);
    p=strsplit(p,"e")[[1]];
    label=paste0('italic(P)~"="~',as.integer(p[1]),'~"\u00d7"~10^',p[2])
  }else if(p>0.1){
    p=sprintf("%.2g",p);
    label=paste0('italic(P)~"="~',p)
  }else{
    p=sprintf("%.3f",p);
    label=paste0('italic(P)~"="~',p)
  }
  return(label)
}
ggplot(dmi_data, aes(x = Group, y = DMI, fill = Group)) +
  geom_violin(aes(fill = Group, 
                  color = Group, 
                  color = after_scale(alpha(color, 0.4))),
              alpha = 0.1, linewidth = 1.2) +
  geom_boxplot(width = 0.05, outlier.shape = 21, outlier.fill = "white", outlier.color = "black", alpha = 0.8) +
  geom_point(aes(color = Group), alpha = 0.4, 
             position = position_jitter(width = 0.1), size = 2.5, stroke = 1.5) +
  geom_errorbar(data = quantiles, aes(x = Group, y = median,
                                      ymin = q25, ymax = q75,
                                      color = Group),
                width = 0.25, size = 1.3) +
  geom_crossbar(data = quantiles, aes(x = Group, y = median, 
                                      ymin = median, ymax = median,
                                      color = Group), 
                width = 0.6, size = 1) +
  stat_compare_means(method = "t.test", label = "p.format", label.y = max(dmi_data$DMI) + 0.1) +
  geom_signif(comparisons = list(c("Low nutrient", "High nutrient")), 
              map_signif_level = map_signif_level,
              y_position = c(2.2), 
              textsize = 4, tip_length = 0.02,
              parse = TRUE) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = c("Low nutrient" = "#B7B5A0", "High nutrient" = "#44757A")) +
  scale_color_manual(values = c("Low nutrient" = "#B7B5A0", "High nutrient" = "#44757A")) +
  theme_test(base_size = 20) +
  labs(title = "DMI Value Distribution (Violin + Boxplot)", y = "DMI Value", x = "Nutrient Level") +
  theme(panel.border = element_rect(color = "black", linewidth = 1),
        strip.background = element_rect(fill = "white", color = "white"),
        axis.text.x = element_text(angle = 0, size = 15, color = "black"),
        axis.text.y = element_text(size = 15, color = "black"),
        legend.position = "none")

dev.off() 
