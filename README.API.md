# AML Station's Marketplace API

Our web API allows users to download datasets and models programatically.

In order to use our API, you will need to

1. [Create an account](https://marketplace.amlstation.com/register) on [log in](https://marketplace.amlstation.com/login) to AutoML Station's Marketplace;
2. Access the [API page](https://marketplace.amlstation.com/api) and click on the **Generate Token** button;

    ![Generate Token button](assets/generate-token.png "Generate Token button")
3. Copy the resulting Token (make sure to save it in a secure place);

    ![Copying the generated token](assets/copying-generated-token.png "Copying the generated token")
4. Paste the Token to the `BEARER_TOKEN` variable in notebook cells;

    ![Paste token to notebook cells](assets/paste-token-to-cells.png "Paste token to notebook cells")

## Finding the correct API URL for datasets and models

Our API URLs have the following structure:

- https://marketplace.amlstation.com/api/v1/ **RESOURCE_TYPE** / **RESOURCE_ID** /download

Where the **RESOURCE_TYPE** is either `models` or `datasets`, and the **RESOURCE_ID** can be found in its webpage URL.

![Finding the resource type and id](assets/resource-type-id.png "Finding the resource type and id")