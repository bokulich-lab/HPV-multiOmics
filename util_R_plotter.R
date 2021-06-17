library("dplyr")
library("tidyverse")
library("ggpubr")
library("cowplot")
library("magrittr")

# Function plotting PC(o)A of in grouped scatter plot
plot_pca_scatters <- function(beta_div2_choose, df_pca_all,
  df_explained_var, ls_targets, output_dir) {

  ls_omics <- c("Microbiome", "Metabolome", "Immunoproteome")
  ls_all_omics <- list()
  i <- 1
  ls_max_omics <- list(-1000, -1000, -1000) %>% set_names(ls_omics)
  ls_min_omics <- list(1000, 1000, 1000) %>% set_names(ls_omics)
  # create omics plots
  for (target in ls_targets) {
    if (target == "T_lactobacillus_dominance") {
      target_str <- "T_lactobacillus_\ndominance"
    } else {
      target_str <- target
    }

    # get explained variance
    for (omics in ls_omics) {
      if (omics == "Microbiome") {
        explained_var_col <- paste0(omics, "_", beta_div2_choose)
      } else {
        explained_var_col <- omics
      }

      vec_pca_values <- df_explained_var %>%
        pull(explained_var_col)

      df2plot <- df_pca_all %>%
        filter(Omics == omics)

      sc_plot <- df2plot %>%
        select(target_col = target, PC1, PC2) %>%
        ggplot() +
        aes(x = PC1, y = PC2, color = target_col) +
        # Compute data ellipses - normal (norm), t-distribution (t) or euclid
        stat_ellipse(type = "norm", level = 0.95) +
        geom_point(alpha = 0.5) +
        { if (i > 9) xlab(paste0("PC1: ",
          round(vec_pca_values[1] * 100, 2), " %"))
        } +
        { if (i <= 9) xlab("")
        } +
        ylab(paste0("PC2: ", round(vec_pca_values[2] * 100, 2), " %")) +
        theme_light() +
        # scale_color_brewer(palette='Set2') +
        { if (i < 4) ggtitle(omics)
        } +
        # rename legend title to actual class name
        labs(color = target_str)

      x_min_plot <- ggplot_build(sc_plot)$layout$panel_params[[1]]$x.range[1]
      x_max_plot <- ggplot_build(sc_plot)$layout$panel_params[[1]]$x.range[2]
      if (x_min_plot < ls_min_omics[[omics]]) {
        ls_min_omics[[omics]] <- x_min_plot
      }
      if (x_max_plot > ls_max_omics[[omics]]) {
        ls_max_omics[[omics]] <- x_max_plot
      }

      ls_all_omics[[i]] <- sc_plot
      i <- i + 1
    }
  }

  # for each x in ls_all_omics remove legend
  ls_all_omics_wo_legend <- map2(ls_all_omics, rep(ls_omics, times = 4), ~ .x +
    theme(legend.position = "none") +
    xlim(c(ls_min_omics[[.y]], ls_max_omics[[.y]])))

  # scale x_lim for each plot

  ls_legends <- map(seq(3, 12, 3), ~ get_legend(ls_all_omics %>%
    extract2(.x) +
    theme(legend.box.margin = margin(0, 10, 0, 0))))

  # plot_grid with omics
  grid_omics <- plot_grid(plotlist = ls_all_omics_wo_legend,
                          nrow = 4, byrow = TRUE,
                          rel_heights = c(1, rep(0.91, 3)))
  grid_legends <- plot_grid(plotlist = ls_legends,
                            nrow = 4, byrow = TRUE,
                            align = "hv")

  grid_both <- plot_grid(grid_omics, grid_legends, ncol = 2,
                        byrow = TRUE, rel_widths = c(6, 1))
  ggsave(
    file = file.path(output_dir, "omics-pca-ggplot.png"),
    dpi = "retina", height = 10, width = 12
  )
}