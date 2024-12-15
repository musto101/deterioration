install.packages("essentials_for_daniel/adnimerge_package/ADNIMERGE_0.0.1.tar.gz",
                 repos = NULL, type = "source")

library(tidyverse)
library(ADNIMERGE)
# library(ggthemes)
# library(naniar)
# library(GGally)

adni <- ADNIMERGE::adnimerge  # load in the adnimerge library

adni_last <- adni %>%
  drop_na(DX) %>%
  filter(COLPROT == 'ADNI2') %>% # using ADNI2 for this project.
  group_by(PTID) %>%
  arrange(M) %>%
  summarise(M = as.numeric(last(M))) # finding the last visit - number of months.

adni_essentials <- adni %>%
  filter(COLPROT == 'ADNI2') %>%
  drop_na(DX) %>%
  mutate(M = as.numeric(M)) %>%
  select(DX, PTID, M) # isolating the essential final DX information in order to do the join for last visit.

adni_last_measure <- adni_last %>%
  left_join(adni_essentials) %>%
  drop_na(DX) %>%
  mutate(last_DX = DX, last_visit = M, PTID = as.character(PTID)) %>%
  select(-DX, -M) # joining such that each p has their final diagnosis and last visit associated with their id.

adni_bl <- read_csv('data/adni_bl.csv') # reading in a preprocessed dat that simply isolates only data from p's first visit.

adni_long <- adni_last_measure %>%
  left_join(adni_bl) %>%
  filter(COLPROT == 'ADNI2')  # then joining with the last visit info table we created.

#table(adni_long$DX.bl) Sanity check
adni_long$X1 <- NULL # removing artifacts
adni_long$...1 <- NULL

adni_wo_missing <- adni_long %>%
  purrr::discard(~ sum(is.na(.x))/length(.x) * 100 >=90) # removing columns with missing data > 90%

adni_slim <- adni_wo_missing %>%
  select(-RID, -PTID, -VISCODE, -SITE, -COLPROT, -ORIGPROT, -EXAMDATE,
         -DX.bl, -FLDSTRENG, -FSVERSION, -IMAGEUID, -FLDSTRENG.bl, DX,
         -FSVERSION.bl, -Years.bl, -Month.bl, -Month, -M, ICV,
         -ends_with('.bl')) # removing unhelpful or superfluous columns

adni_slim$last_DX <- as.character(adni_slim$last_DX) # making sure last DX is a character and not a factor

adni_slim$ABETA <- as.numeric(gsub("[^0-9.]", "", adni_slim$ABETA)) # Standardising the expression of csf derived information. THIS IS REMOVED IN THE OTHER DATASET.
adni_slim$TAU <- as.numeric(gsub("[^0-9.]", "", adni_slim$TAU))
adni_slim$PTAU <- as.numeric(gsub("[^0-9.]", "", adni_slim$PTAU))

write.csv(adni_slim, 'data/adni2_slim.csv') # writing to csv file

