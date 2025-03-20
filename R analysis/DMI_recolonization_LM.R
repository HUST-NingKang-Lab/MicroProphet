library(dplyr)

dmi_results_all <- read.csv("dmi_results.csv")

dmi_results_all$DMI <- log10(dmi_results_all$DMI + 1e-5)

recolonization_rates <- read.csv("recolonization_rates.csv")

genera_recolonization <- recolonization_rates$Genus

genus_list <- dmi_results_all$genus

filtered_recolonization_rates <- recolonization_rates %>%
  filter(Genus %in% genus_list)
genus_list <- filtered_recolonization_rates$Genus
filtered_dmi_results_all <- dmi_results_all %>% filter(Genus %in% genus_list)
filtered_recolonization_rates$Recolonization_Rate <- log10(filtered_recolonization_rates$Recolonization_Rate + 1e-5)
data = cbind(dmi_results_all, recolonization_rates)
#write.csv(data, file = "DMI_recolonization.csv")
correlation <- cor(data$Normalized_Rank, data$DMI, method = "pearson")
correlation
lm_model <- lm(DMI ~ average_abundance, data = data)
summary_lm <- summary(lm_model)

p_value <- summary_lm$coefficients[2, "Pr(>|t|)"]
r_squared_lm <- summary_lm$r.squared  
pdf("DMI_Recolonization.pdf", width = 6.5, height = 6)
ggplot(data, aes(x = Recolonization_Rate, y = DMI)) +
  
  geom_point(shape = 21, color = "black", fill = "#549F9A", size = 3, stroke = 1) +
  
  geom_abline(linetype = "dashed", color = "black", size = 1) +
  
  geom_smooth(method = "lm", se = FALSE, color = "#549F9A", size = 2) +
  
  labs(
    title = "",
    x = "Recolonization Rate",
    y = "DMI"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, size = 1), 
    plot.margin = unit(c(1, 1, 1, 1), "cm")
  ) +
  
  annotate("text", 
           label = paste("P =", formatC(p_value, format = "e", digits = 2),
                         ", Marg. R² =", signif(r_squared_lm, digits = 3)),
           size = 5, hjust = 0, color = "black") +
  
  annotate("text", 
           label = paste("Avg. residual =", signif(mean(abs(data$DMI - data$Recolonization_Rate)), digits = 3)),
           size = 5, hjust = 0, color = "black")


dev.off()
