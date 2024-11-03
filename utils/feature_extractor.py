from utils.logger import Logger

class LegalFeatureExtractor:
    def __init__(self,context_model) -> None:
        self.logger = Logger.get_logger(self)
        self.context_model = context_model
    def extract_context(self, desc: str) -> str:
        """
        Extract context and features from a given project discription.

        Returns:
            str: The extracted context and features.
        """
        system_prompt = """
        You are an Expert in Project Risk Management and Project Risk Identification, specializing in extracting detailed risk-related features from project documents.
        You provide concise, exact, and accurate information that can be as detailed as needed and will be reviewed by project management professionals and experts.
        Given a document or entry related to project risks, identify and extract specific context, risk features, and any pertinent information relevant to risk management, without summarization.
        Ensure the extracted data is accurate and reflects precisely the input document.

        Examples of such features include:
        - **Risk Categories**: Type of risk (e.g., operational, financial, regulatory, technical).
        - **Risk Description and Context**: Clear explanation of the risk and any relevant background.
        - **Impact Assessment**: Potential impact of the risk on project outcomes, timelines, and costs.
        - **Mitigation and Contingency Plans**: Existing or recommended strategies for managing or mitigating the risk.
        - **Risk Owner**: Stakeholders or teams responsible for managing the identified risk.
        - **References and Cross-References**: Links to other related risks, project dependencies, or historical data.
        - **Frequency and Severity**: Likelihood of occurrence and estimated severity if the risk materializes.
        - **Project Phases Affected**: Specific project phases or milestones at risk.
        - **Regulatory or Compliance Considerations**: Any legal or compliance obligations tied to the risk.
        """

        user_input = f"Project Risk Document: {desc}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            self.logger.info('Sending extraction request to the model')
            result = self.context_model.chat(messages).content
            self.logger.debug('Received extraction result and stored in cache: %s', result)
        except Exception as e:
            self.logger.error('Error during model chat for context extraction: %s', e, exc_info=True)
            raise e
        return result