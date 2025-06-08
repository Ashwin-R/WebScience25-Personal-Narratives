library("lme4")
library("emmeans")
library("glmmTMB")
library(lmerTest)
library("stargazer")

df <- read.csv("../../data/final_h1.csv")
df$subreddit <- as.factor(df$subreddit)
df$month <- as.factor(df$month)
df$quartile <- as.factor(df$quartile)


df$quartile <- relevel(df$quartile, ref = "top 25%")

h3_model <- lmer(cbrt_score ~  + (1|subreddit) + (1|month) + is_top_level + is_corona + cbrt_post_score + quartile*is_narrative, data=df)

summary(h3_model)
updGrid <- update(ref_grid(h3_model), 
                  tran = make.tran("power", alpha = 1/3,
                                   beta = 0))
h3_model_em <- emmeans(updGrid,pairwise ~ quartile*is_narrative, type = "response")

h3_model_em

# # Convert the model summary to LaTeX format using stargazer
# stargazer(h3_model, type = "text", title = "Generalized Linear Mixed Model Results", 
#           out = "glmm_results_RQ.tex", digits = 3)
em_m3 <- emmeans(h3_model, pairwise ~ is_narrative*quartile,  pbkrtest.limit = 24000, lmerTest.limit = 6000,type = "response")
em_m3