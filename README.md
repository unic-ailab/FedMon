# FedMon: Federated Learning Monitoring Toolkit

## Overview
FedMon is a comprehensive toolkit designed to facilitate the monitoring of Federated Learning (FL) deployments. It aims to simplify the process of extracting FL-specific and system-level metrics, aligning metrics to training rounds, pinpointing performance inefficiencies, and comparing current to previous deployments.

## Features
- **Seamless Integration**: Integrates probing interface with FL deployment, automating metric extraction.
- **Rich Metric Set**: Provides system, dataset, model, and experiment-level metrics with configurable periodicity.
- **Analytic Tool**: Includes a library for post-experimentation exploratory data analysis and comparative studies.
- **User Interface**: Employs Jupyter Notebooks for interactive and exploratory computing.

## Getting Started
1. **Prerequisites**: Ensure you have Docker installed for containerized FL services and Jupyter Notebook for data analysis.
2. **Installation**: Clone this repository and build the Docker images provided for FedMon clients and server.
3. **Configuration**: Customize the FedMon configuration files to suit your specific FL setup (number of clients, rounds, etc.).

## Usage
- **Running FL Services**: Deploy the FedMon-enhanced FL services in your containerized environment.
- **Monitoring**: Utilize the FedMon probes to collect metrics throughout the FL process.
- **Analysis**: Leverage the FedMon Analysis Library in Jupyter Notebook to analyze collected metrics.

## Example Use Case
- Describes an edge-based distributed learning system for handwriting recognition using IoT devices, with a focus on the MNIST dataset and CNN models.

## Contribution
- Contributions to FedMon are welcome. Please refer to the contribution guidelines for more details.

## License
- This project is licensed under the terms of the Apache License.

## Acknowledgments
- Authors: Moysis Symeonides, Demetris Trihinas, and Fotis Nikolaidis.
- This research received no external funding.

## Contact
For further inquiries, please contact:
- Moysis Symeonides at symeonidis.moysis@ucy.ac.cy
- Demetris Trihinas at trihinas.d@unic.ac.cy
- Fotis Nikolaidis at fotis@superduperdb.com