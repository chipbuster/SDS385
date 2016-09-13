# Calculate w_i for some sample x and parameter beta
genWeight <- function(x,beta){
    exponent <- -sum(x * beta)  # Generate exponent by taking neg dot product

    return (1 / (1 + exp(exponent)))   #R returns are weird o.0
}

## Generates numSamples draws from the distribution described in
# the homework handout with parameter beta. Assumes single-trial
# distributions.
genSamples <- function(beta, numSamples){
    datadim <- length(beta)

    df <- data.frame(matrix(ncol = datadim + 1, nrow = 0))
    
    for (j in 1:numSamples){

      sample <- rexp(datadim, rate = 0.5)
      weight <- genWeight(sample, beta)

      # 0/1 outcome, one sample, with probabilty given by weight
      outcome <- rbinom(1, 1, weight)
      
      datarow <- c(outcome, sample)
      df <- rbind(df, datarow)
    }

    return(df)
}

# Sane default choices for beta, create an output
genData <- function(dim){
  beta <- rnorm(dim)

  fname <- paste(beta,collapse="=")
  fname <- paste(fname,"csv", sep=".")
  
  data <- genSamples(beta, 10000)

  write.table(data, file=fname, sep=",", row.names=FALSE, col.names=FALSE)
  print(mean(data[,1]))
}
