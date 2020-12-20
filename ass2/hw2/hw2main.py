#!/usr/bin/env python3
"""
hw2main.py

UNSW COMP9444 Neural Networks and Deep Learning

DO NOT MODIFY THIS FILE
"""

import torch
from torchtext import data

from config import device
import student

def main():
    print("Using device: {}"
          "\n".format(str(device)))

    # Load the training dataset, and create a dataloader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           tokenize=student.tokenise,
                           preprocessing=student.preprocessing,
                           postprocessing=student.postprocessing,
                           stop_words=student.stopWords)
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField),
                                  'businessCategory': ('businessCategory', labelField)})

    textField.build_vocab(dataset, vectors=student.wordVectors)

    # Allow training on the entire dataset, or split it for training and validation.
    if student.trainValSplit == 1:
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                          batch_size=student.batchSize,
                                          sort_key=lambda x: len(x.reviewText),
                                          sort_within_batch=True)
    else:
        train, validate = dataset.split(split_ratio=student.trainValSplit)

        trainLoader, valLoader = data.BucketIterator.splits((train, validate),
                                                            shuffle=True,
                                                            batch_size=student.batchSize,
                                                            sort_key=lambda x: len(x.reviewText),
                                                            sort_within_batch=True)

    # Get model and optimiser from student.
    net = student.net.to(device)
    lossFunc = student.lossFunc
    optimiser = student.optimiser

    # Train.
    for epoch in range(student.epochs):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            rating = batch.rating.to(device)
            businessCategory = batch.businessCategory.to(device)

            # PyTorch calculates gradients by accumulating contributions to them
            # (useful for RNNs).  Hence we must manually set them to zero before
            # calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            ratingOutput, categoryOutput = net(inputs, length)
            loss = lossFunc(ratingOutput, categoryOutput, rating, businessCategory)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            runningLoss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                      % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0

    # Save model.
    torch.save(net.state_dict(), 'savedModel.pth')
    print("\n"
          "Model saved to savedModel.pth")

    # Test on validation data if it exists.
    if student.trainValSplit != 1:
        net.eval()

        correctRatingOnlySum = 0
        correctCategoryOnlySum = 0
        bothCorrectSum = 0
        with torch.no_grad():
            for batch in valLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                rating = batch.rating.to(device)
                businessCategory = batch.businessCategory.to(device)

                # Convert network output to integer values.
                ratingOutputs, categoryOutputs = student.convertNetOutput(*net(inputs, length))

                # Calculate performance
                correctRating = rating == ratingOutputs.flatten()
                correctCategory = businessCategory == categoryOutputs.flatten()

                correctRatingOnlySum += torch.sum(correctRating & ~correctCategory).item()
                correctCategoryOnlySum += torch.sum(correctCategory & ~correctRating).item()
                bothCorrectSum += torch.sum(correctRating & correctCategory).item()

        correctRatingOnlyPercent = correctRatingOnlySum / len(validate)
        correctCategoryOnlyPercent = correctCategoryOnlySum / len(validate)
        bothCorrectPercent = bothCorrectSum / len(validate)
        neitherCorrectPer = 1 - correctRatingOnlyPercent \
                              - correctCategoryOnlyPercent \
                              - bothCorrectPercent

        score = 100 * (bothCorrectPercent
                       + 0.5 * correctCategoryOnlyPercent
                       + 0.1 * correctRatingOnlyPercent)

        print("\n"
              "Rating incorrect, business category incorrect: {:.2%}\n"
              "Rating correct, business category incorrect: {:.2%}\n"
              "Rating incorrect, business category correct: {:.2%}\n"
              "Rating correct, business category correct: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}".format(neitherCorrectPer,
                                              correctRatingOnlyPercent,
                                              correctCategoryOnlyPercent,
                                              bothCorrectPercent, score))

if __name__ == '__main__':
    main()
