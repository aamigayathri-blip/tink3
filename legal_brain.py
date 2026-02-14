import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Specialized Data: Sample of the new Bharatiya Nyaya Sanhita (BNS) 2023
# In production, this would be loaded from a secure D

BNS_DATA = [
    {
        "id": 1,
        "section": "BNS Section 316",
        "title": "Criminal Breach of Trust",
        "text": "Whoever, being in any manner entrusted with property, dishonestly misappropriates or converts to his own use that property, commits criminal breach of trust.",
        "simplified": "If you gave someone money or items to hold and they used it for themselves, this is a crime."
    },
    {
        "id": 2,
        "section": "BNS Section 318",
        "title": "Cheating",
        "text": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person...",
        "simplified": "If someone lied to you to get your money or property (Fraud), it falls under this section."
    },
    {
        "id": 3,
        "section": "BNS Section 69",
        "title": "Sexual Intercourse by Deceitful Means",
        "text": "Whoever, by deceitful means or making promise to marry to a woman without any intention of fulfilling the same, has sexual intercourse...",
        "simplified": "False promises of marriage to gain intimacy are now strictly punishable under BNS."
    },
    
    {
        "id": 4,
        "section": "BNS Section 103",
        "title": "Punishment for Murder",
        "text": "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
        "simplified": "This section replaces the old IPC 302. It defines the punishment for intentionally killing a person."
    },
    {
        "id": 5,
        "section": "BNS Section 103(2)",
        "title": "Mob Lynching",
        "text": "When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief...",
        "simplified": "A new specific law punishing groups of 5 or more people who kill someone based on religion, race, or caste."
    },
    {
        "id": 6,
        "section": "BNS Section 303",
        "title": "Theft",
        "text": "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.",
        "simplified": "Taking someone's movable belongings (like a phone or wallet) without their permission."
    },
    {
        "id": 7,
        "section": "BNS Section 304",
        "title": "Snatching",
        "text": "Theft is 'snatching' if, in order to commit theft, the offender suddenly or quickly or forcibly seizes or secures or grabs or takes away from any person any movable property.",
        "simplified": "This differentiates 'snatching' (like chain or purse snatching) from general theft, acknowledging the sudden or forceful nature of the act."
    },
    {
        "id": 8,
        "section": "BNS Section 356",
        "title": "Defamation",
        "text": "Whoever, by words either spoken or intended to be read, or by signs or by visible representations, makes or publishes any imputation concerning any person intending to harm...",
        "simplified": "Ruining someone's reputation publicly using spoken words, writing, or images."
    },
    {
        "id": 9,
        "section": "BNS Section 111",
        "title": "Organised Crime",
        "text": "Any continuing unlawful activity including kidnapping, robbery, vehicle theft, extortion, land grabbing, contract killing, economic offence, cyber-crimes... carried out by a group.",
        "simplified": "A new, tough section targeting crime syndicates and mafias involved in organized illegal activities."
    },
    {
        "id": 10,
        "section": "BNS Section 351",
        "title": "Criminal Intimidation",
        "text": "Whoever threatens another with any injury to his person, reputation or property, or to the person or reputation of any one in whom that person is interested...",
        "simplified": "Threatening to hurt someone or damage their property to force them to do something against their will."
    },
    
    {
        "id": 11,
        "section": "BNS Section 113",
        "title": "Terrorist Act",
        "text": "Whoever does any act with the intent to threaten or likely to threaten the unity, integrity, sovereignty, security, or economic security of India or with the intent to strike terror in the people...",
        "simplified": "A comprehensive section defining terrorism. It covers acts that threaten the country's safety, economic stability, or cause public fear."
    },
    {
        "id": 12,
        "section": "BNS Section 152",
        "title": "Act Endangering Sovereignty (Replaces Sedition)",
        "text": "Whoever, purposely or knowingly, by words, either spoken or written, or by signs, or by visible representation... excites or attempts to excite, secession or armed rebellion or subversive activities...",
        "simplified": "This replaces the old 'Sedition' law (IPC 124A). It specifically punishes acts that encourage breaking away from India (secession) or armed rebellion."
    },
    {
        "id": 13,
        "section": "BNS Section 64",
        "title": "Punishment for Rape",
        "text": "A man is said to commit 'rape' if he... has sexual intercourse with a woman under circumstances falling under any of the seven descriptions... without her consent.",
        "simplified": "Defines the crime of rape and sets strict punishments (formerly IPC 375/376). Minimum imprisonment has often been increased compared to older laws."
    },
    {
        "id": 14,
        "section": "BNS Section 79",
        "title": "Dowry Death",
        "text": "Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage and it is shown that she was subjected to cruelty...",
        "simplified": "If a woman dies unnaturally within 7 years of marriage due to harassment over dowry, the husband or relatives are held responsible."
    },
    {
        "id": 15,
        "section": "BNS Section 281",
        "title": "Rash Driving",
        "text": "Whoever drives any vehicle on any public way in a manner so rash or negligent as to endanger human life, or to be likely to cause hurt or injury to any other person...",
        "simplified": "Driving recklessly or negligently on public roads in a way that risks other people's lives (Formerly IPC 279)."
    },
    {
        "id": 16,
        "section": "BNS Section 109",
        "title": "Attempt to Murder",
        "text": "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty of murder...",
        "simplified": "Trying to kill someone but failing. If you hurt them in the process, the punishment is more severe (Formerly IPC 307)."
    },
    {
        "id": 17,
        "section": "BNS Section 74",
        "title": "Sexual Harassment",
        "text": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will there by outrage her modesty...",
        "simplified": "Unwanted physical contact, demands for sexual favors, or making sexually coloured remarks (Formerly IPC 354)."
    },
    {
        "id": 18,
        "section": "BNS Section 309",
        "title": "Robbery",
        "text": "In all robbery there is either theft or extortion. Theft is 'robbery' if, in order to the committing of the theft... the offender, for that end, voluntarily causes or attempts to cause to any person death or hurt...",
        "simplified": "Theft becomes 'robbery' when violence, or the threat of violence/death, is used to steal the items."
    },
    {
        "id": 19,
        "section": "BNS Section 329",
        "title": "Criminal Trespass",
        "text": "Whoever enters into or upon property in the possession of another with intent to commit an offence or to intimidate, insult or annoy any person in possession of such property...",
        "simplified": "Entering someone else's property without permission to commit a crime or to harass/threaten the people living there."
    },
    {
        "id": 20,
        "section": "BNS Section 226",
        "title": "Attempt to Commit Suicide to Compel Public Servant",
        "text": "Whoever attempts to commit suicide with the intent to compel or restrain any public servant from discharging his official duty...",
        "simplified": "Unlike the old IPC 309 (which criminalized all suicide attempts), BNS focuses on punishing those who use suicide attempts specifically to blackmail public officials or stop them from doing their job."
    },
    {
        "id": 21,
        "section": "BNS Section 115",
        "title": "Voluntarily Causing Hurt",
        "text": "Whoever causes bodily pain, disease or infirmity to any person is said to cause hurt.",
        "simplified": "Physically injuring someone intentionally, even if the injury is minor."
    },
    {
        "id": 22,
        "section": "BNS Section 117",
        "title": "Voluntarily Causing Grievous Hurt",
        "text": "Whoever voluntarily causes grievous hurt shall be punished with imprisonment and fine.",
        "simplified": "Causing serious injuries like fractures, permanent damage, or loss of sight/hearing."
    },
    {
        "id": 23,
        "section": "BNS Section 121",
        "title": "Wrongful Restraint",
        "text": "Whoever voluntarily obstructs any person so as to prevent that person from proceeding in any direction in which they have a right to proceed.",
        "simplified": "Blocking someone’s path illegally so they cannot move freely."
    },
    {
        "id": 24,
        "section": "BNS Section 122",
        "title": "Wrongful Confinement",
        "text": "Whoever wrongfully restrains any person in such a manner as to prevent that person from proceeding beyond certain limits.",
        "simplified": "Illegally locking or keeping someone confined in a space."
    },
    {
        "id": 25,
        "section": "BNS Section 140",
        "title": "Kidnapping",
        "text": "Whoever takes or entices any minor or person of unsound mind out of the keeping of the lawful guardian without consent.",
        "simplified": "Taking a child or mentally unsound person away from their guardian without permission."
    },
    {
        "id": 26,
        "section": "BNS Section 141",
        "title": "Abduction",
        "text": "Whoever by force compels, or by deceitful means induces, any person to go from any place.",
        "simplified": "Forcibly or fraudulently taking someone away from one place to another."
    },
    {
        "id": 27,
        "section": "BNS Section 191",
        "title": "Giving False Evidence",
        "text": "Whoever legally bound by oath makes a false statement which he knows to be false.",
        "simplified": "Lying in court after taking an oath."
    },
    {
        "id": 28,
        "section": "BNS Section 194",
        "title": "Fabricating False Evidence",
        "text": "Whoever causes any circumstance to exist intending that it may appear in evidence and mislead the court.",
        "simplified": "Creating fake proof to mislead a judge or court."
    },
    {
        "id": 29,
        "section": "BNS Section 204",
        "title": "Destruction of Evidence",
        "text": "Whoever causes disappearance of evidence of an offence with intent to screen the offender.",
        "simplified": "Destroying or hiding proof to protect a criminal."
    },
    {
        "id": 30,
        "section": "BNS Section 233",
        "title": "Counterfeiting Currency",
        "text": "Whoever counterfeits or knowingly performs any part of the process of counterfeiting currency-notes.",
        "simplified": "Making fake money or being involved in producing fake currency."
    },
    {
        "id": 31,
        "section": "BNS Section 242",
        "title": "Criminal Misappropriation of Property",
        "text": "Whoever dishonestly misappropriates or converts to his own use any movable property.",
        "simplified": "Using someone else's lost or entrusted property for your own benefit."
    },
    {
        "id": 32,
        "section": "BNS Section 270",
        "title": "Public Nuisance",
        "text": "Whoever does any act or is guilty of illegal omission which causes common injury, danger or annoyance to the public.",
        "simplified": "Doing something illegal that harms or disturbs the public."
    },
    {
        "id": 33,
        "section": "BNS Section 294",
        "title": "Obscene Acts",
        "text": "Whoever, to the annoyance of others, does any obscene act in any public place.",
        "simplified": "Performing obscene actions in public places."
    },
    {
        "id": 34,
        "section": "BNS Section 298",
        "title": "Injuring Religious Feelings",
        "text": "Whoever deliberately and maliciously outrages the religious feelings of any class by insulting its religion.",
        "simplified": "Intentionally insulting someone’s religion to provoke anger."
    },
    {
        "id": 35,
        "section": "BNS Section 310",
        "title": "Extortion",
        "text": "Whoever intentionally puts any person in fear of injury and thereby dishonestly induces the person to deliver property.",
        "simplified": "Forcing someone to give money or property by threatening them."
    },
    {
        "id": 36,
        "section": "BNS Section 311",
        "title": "Extortion by Threat of Death",
        "text": "Whoever commits extortion by putting a person in fear of death or grievous hurt.",
        "simplified": "Demanding money by threatening to kill or seriously injure someone."
    },
    {
        "id": 37,
        "section": "BNS Section 312",
        "title": "Dishonestly Receiving Stolen Property",
        "text": "Whoever dishonestly receives or retains any stolen property knowing or having reason to believe it to be stolen.",
        "simplified": "Keeping or buying stolen goods while knowing they are stolen."
    },
    {
        "id": 38,
        "section": "BNS Section 319",
        "title": "Cheating by Personation",
        "text": "Whoever cheats by pretending to be some other person or by knowingly substituting one person for another.",
        "simplified": "Impersonating someone else to cheat people."
    },
    {
        "id": 39,
        "section": "BNS Section 320",
        "title": "Mischief",
        "text": "Whoever causes destruction of property or change in it with intent to cause wrongful loss or damage.",
        "simplified": "Damaging someone’s property intentionally."
    },
    {
        "id": 40,
        "section": "BNS Section 321",
        "title": "Mischief by Fire",
        "text": "Whoever commits mischief by fire or explosive substance intending to destroy property.",
        "simplified": "Setting fire to property to cause destruction."
    },
    {
        "id": 41,
        "section": "BNS Section 324",
        "title": "House Trespass",
        "text": "Whoever commits criminal trespass by entering into any building used as a human dwelling.",
        "simplified": "Illegally entering someone’s house or building."
    },
    {
        "id": 42,
        "section": "BNS Section 330",
        "title": "House Breaking",
        "text": "Whoever commits house trespass after breaking open a lock or using force.",
        "simplified": "Breaking into someone’s house illegally."
    },
    {
        "id": 43,
        "section": "BNS Section 333",
        "title": "Forgery",
        "text": "Whoever makes any false document with intent to cause damage or support any claim.",
        "simplified": "Creating fake documents to cheat or harm someone."
    },
    {
        "id": 44,
        "section": "BNS Section 335",
        "title": "Forgery for Cheating",
        "text": "Forgery committed for the purpose of cheating.",
        "simplified": "Making fake documents specifically to cheat someone."
    },
    {
        "id": 45,
        "section": "BNS Section 338",
        "title": "Using Forged Document",
        "text": "Whoever fraudulently uses as genuine any forged document.",
        "simplified": "Using fake documents as if they were real."
    },
    {
        "id": 46,
        "section": "BNS Section 340",
        "title": "Falsification of Accounts",
        "text": "Whoever willfully falsifies or alters financial accounts.",
        "simplified": "Changing financial records to hide fraud."
    },
    {
        "id": 47,
        "section": "BNS Section 343",
        "title": "Criminal Conspiracy",
        "text": "When two or more persons agree to do an illegal act or a legal act by illegal means.",
        "simplified": "Planning a crime together."
    },
    {
        "id": 48,
        "section": "BNS Section 348",
        "title": "Unlawful Assembly",
        "text": "An assembly of five or more persons with a common illegal objective.",
        "simplified": "Group of 5 or more people gathering with illegal intent."
    },
    {
        "id": 49,
        "section": "BNS Section 349",
        "title": "Rioting",
        "text": "When force or violence is used by an unlawful assembly.",
        "simplified": "Violent behavior by a group."
    },
    {
        "id": 50,
        "section": "BNS Section 350",
        "title": "Armed Rioting",
        "text": "Rioting while armed with a deadly weapon.",
        "simplified": "Participating in a riot while carrying weapons."
    },
    {
        "id": 51,
        "section": "BNS Section 352",
        "title": "Assault",
        "text": "Whoever makes any gesture intending to cause another person to fear criminal force.",
        "simplified": "Threatening physical harm through actions or gestures."
    },
    {
        "id": 52,
        "section": "BNS Section 353",
        "title": "Assault on Public Servant",
        "text": "Assaulting or using criminal force against a public servant during duty.",
        "simplified": "Attacking a government official while they are working."
    },
    {
        "id": 53,
        "section": "BNS Section 357",
        "title": "Breach of Peace",
        "text": "Acts that disturb public tranquility.",
        "simplified": "Creating chaos or disturbance in public."
    },
    {
        "id": 54,
        "section": "BNS Section 360",
        "title": "Kidnapping for Ransom",
        "text": "Kidnapping a person and demanding ransom for their release.",
        "simplified": "Abducting someone to demand money."
    },
    {
        "id": 55,
        "section": "BNS Section 361",
        "title": "Human Trafficking",
        "text": "Recruitment, transport, harboring or receipt of persons for exploitation.",
        "simplified": "Illegally moving or controlling people for forced labor or exploitation."
    },
    {
        "id": 56,
        "section": "BNS Section 362",
        "title": "Child Trafficking",
        "text": "Trafficking involving minors for exploitation.",
        "simplified": "Buying, selling, or exploiting children illegally."
    },
    {
        "id": 57,
        "section": "BNS Section 370",
        "title": "Harboring Offender",
        "text": "Whoever shelters or protects a person who has committed an offence.",
        "simplified": "Helping a criminal hide from the police."
    },
    {
        "id": 58,
        "section": "BNS Section 372",
        "title": "Sale of Minor for Prostitution",
        "text": "Selling or hiring a minor for purposes of prostitution.",
        "simplified": "Selling a child for sexual exploitation."
    },
    {
        "id": 59,
        "section": "BNS Section 373",
        "title": "Buying Minor for Prostitution",
        "text": "Buying or hiring a minor for prostitution.",
        "simplified": "Purchasing a child for exploitation."
    },
    {
        "id": 60,
        "section": "BNS Section 375",
        "title": "Cyber Fraud",
        "text": "Fraud committed using electronic communication or digital platforms.",
        "simplified": "Online scams involving digital transactions or identity theft."
    },
    {
        "id": 61,
        "section": "BNS Section 376",
        "title": "Identity Theft",
        "text": "Fraudulent use of another person's identity for unlawful gain.",
        "simplified": "Using someone else's personal details to commit fraud."
    },
    {
        "id": 62,
        "section": "BNS Section 377",
        "title": "Online Impersonation",
        "text": "Pretending to be another person through electronic means for deception.",
        "simplified": "Creating fake online profiles to deceive others."
    },
    {
        "id": 63,
        "section": "BNS Section 378",
        "title": "Data Theft",
        "text": "Unauthorized access and extraction of confidential digital data.",
        "simplified": "Stealing sensitive digital information."
    },
    {
        "id": 64,
        "section": "BNS Section 379",
        "title": "Unauthorized System Access",
        "text": "Accessing a protected computer system without permission.",
        "simplified": "Hacking into someone’s system illegally."
    },
    {
        "id": 65,
        "section": "BNS Section 380",
        "title": "Electronic Evidence Tampering",
        "text": "Destroying or altering digital evidence with intent to mislead investigation.",
        "simplified": "Deleting or editing digital proof to escape punishment."
    },
    {
        "id": 66,
        "section": "BNS Section 381",
        "title": "Cyber Stalking",
        "text": "Repeated online harassment causing fear or distress.",
        "simplified": "Harassing someone persistently through digital platforms."
    },
    {
        "id": 67,
        "section": "BNS Section 382",
        "title": "Obscene Digital Content",
        "text": "Publishing or transmitting obscene material electronically.",
        "simplified": "Sharing explicit content online illegally."
    },
    {
        "id": 68,
        "section": "BNS Section 383",
        "title": "Defamation Through Electronic Means",
        "text": "Publishing defamatory statements using digital platforms.",
        "simplified": "Spreading false statements online to harm reputation."
    },
    {
        "id": 69,
        "section": "BNS Section 384",
        "title": "Financial Fraud Using Digital Payment",
        "text": "Fraud committed via digital banking or payment systems.",
        "simplified": "Scamming people through online money transfers."
    },
    {
        "id": 70,
        "section": "BNS Section 385",
        "title": "Cryptocurrency Fraud",
        "text": "Fraudulent schemes involving digital or virtual currencies.",
        "simplified": "Cheating people using crypto scams."
    },
    {
        "id": 71,
        "section": "BNS Section 386",
        "title": "Cyber Terrorism",
        "text": "Use of computer resources to threaten national security.",
        "simplified": "Using hacking or digital attacks to threaten the country."
    },
    {
        "id": 72,
        "section": "BNS Section 387",
        "title": "Publishing Fake News",
        "text": "Creating or spreading false information causing public harm.",
        "simplified": "Spreading dangerous fake information online."
    },
    {
        "id": 73,
        "section": "BNS Section 388",
        "title": "Deepfake Manipulation",
        "text": "Creating manipulated digital media to deceive or defame.",
        "simplified": "Using AI to create fake videos or images."
    },
    {
        "id": 74,
        "section": "BNS Section 389",
        "title": "Online Radicalization",
        "text": "Promoting extremist ideology using digital platforms.",
        "simplified": "Using the internet to spread extremist beliefs."
    },
    {
        "id": 75,
        "section": "BNS Section 390",
        "title": "Electronic Blackmail",
        "text": "Threatening to release private digital content for gain.",
        "simplified": "Blackmailing someone using private photos or messages."
    },
    {
        "id": 76,
        "section": "BNS Section 391",
        "title": "Digital Privacy Violation",
        "text": "Unauthorized surveillance or monitoring of digital communication.",
        "simplified": "Spying on someone’s online activity illegally."
    },
    {
        "id": 77,
        "section": "BNS Section 392",
        "title": "Mass Data Breach",
        "text": "Large-scale unauthorized exposure of personal data.",
        "simplified": "Leaking thousands of people’s private data."
    },
    {
        "id": 78,
        "section": "BNS Section 393",
        "title": "Online Gambling Fraud",
        "text": "Running illegal or fraudulent online betting platforms.",
        "simplified": "Cheating people through fake online betting sites."
    },
    {
        "id": 79,
        "section": "BNS Section 394",
        "title": "Phishing Scam",
        "text": "Obtaining sensitive information through deceptive digital communication.",
        "simplified": "Tricking people into giving passwords or bank details."
    },
    {
        "id": 80,
        "section": "BNS Section 395",
        "title": "Ransomware Attack",
        "text": "Locking digital systems and demanding payment for restoration.",
        "simplified": "Hacking a system and demanding money to unlock it."
    },
    {
        "id": 81,
        "section": "BNS Section 396",
        "title": "Digital Evidence Fabrication",
        "text": "Creating false digital records to implicate someone.",
        "simplified": "Planting fake digital proof to frame someone."
    },
    {
        "id": 82,
        "section": "BNS Section 397",
        "title": "Botnet Operation",
        "text": "Operating networks of compromised devices for illegal activities.",
        "simplified": "Controlling hacked devices to launch cyber attacks."
    },
    {
        "id": 83,
        "section": "BNS Section 398",
        "title": "Malware Distribution",
        "text": "Intentionally spreading malicious software.",
        "simplified": "Sending viruses or harmful software to others."
    },
    {
        "id": 84,
        "section": "BNS Section 399",
        "title": "SIM Swap Fraud",
        "text": "Fraudulently transferring a phone number to gain access to accounts.",
        "simplified": "Hijacking someone’s phone number to steal money."
    },
    {
        "id": 85,
        "section": "BNS Section 400",
        "title": "Unauthorized Biometric Use",
        "text": "Using fingerprint, facial or iris data without consent.",
        "simplified": "Misusing someone’s biometric information."
    },
    {
        "id": 86,
        "section": "BNS Section 401",
        "title": "AI Generated Fraud",
        "text": "Using artificial intelligence systems to commit fraud.",
        "simplified": "Using AI tools to cheat or scam people."
    },
    {
        "id": 87,
        "section": "BNS Section 402",
        "title": "Financial Market Manipulation",
        "text": "Using digital means to manipulate stock or crypto markets.",
        "simplified": "Artificially influencing prices for illegal gain."
    },
    {
        "id": 88,
        "section": "BNS Section 403",
        "title": "Dark Web Trade",
        "text": "Engaging in illegal trade through anonymous digital networks.",
        "simplified": "Buying or selling illegal goods online secretly."
    },
    {
        "id": 89,
        "section": "BNS Section 404",
        "title": "Digital Extortion",
        "text": "Demanding money by threatening digital harm or exposure.",
        "simplified": "Threatening to leak data unless paid."
    },
    {
        "id": 90,
        "section": "BNS Section 405",
        "title": "Cloud Data Sabotage",
        "text": "Destroying or corrupting data stored in cloud systems.",
        "simplified": "Damaging online stored files intentionally."
    },
    {
        "id": 91,
        "section": "BNS Section 406",
        "title": "Corporate Data Theft",
        "text": "Stealing confidential business information electronically.",
        "simplified": "Taking company secrets digitally."
    },
    {
        "id": 92,
        "section": "BNS Section 407",
        "title": "Digital Money Laundering",
        "text": "Concealing illicit funds through digital financial systems.",
        "simplified": "Hiding illegal money using online transactions."
    },
    {
        "id": 93,
        "section": "BNS Section 408",
        "title": "Smart Contract Exploitation",
        "text": "Abusing blockchain contracts for illegal gain.",
        "simplified": "Manipulating blockchain code to steal funds."
    },
    {
        "id": 94,
        "section": "BNS Section 409",
        "title": "Online Auction Fraud",
        "text": "Deceiving buyers or sellers in digital auction platforms.",
        "simplified": "Cheating people in online bidding platforms."
    },
    {
        "id": 95,
        "section": "BNS Section 410",
        "title": "Fake Investment Scheme",
        "text": "Operating fraudulent digital investment platforms.",
        "simplified": "Running fake online investment schemes."
    },
    {
        "id": 96,
        "section": "BNS Section 411",
        "title": "Unauthorized Surveillance Software",
        "text": "Installing spyware without consent.",
        "simplified": "Secretly tracking someone using software."
    },
    {
        "id": 97,
        "section": "BNS Section 412",
        "title": "Digital Sabotage of Critical Infrastructure",
        "text": "Attacking essential public digital systems.",
        "simplified": "Hacking systems like power grids or hospitals."
    },
    {
        "id": 98,
        "section": "BNS Section 413",
        "title": "Electronic Document Suppression",
        "text": "Intentionally hiding digital documents required by law.",
        "simplified": "Deleting important digital records to avoid liability."
    },
    {
        "id": 99,
        "section": "BNS Section 414",
        "title": "Cross-Border Cyber Crime",
        "text": "Cyber offence involving multiple countries.",
        "simplified": "Committing digital crime affecting different nations."
    },
    {
        "id": 100,
        "section": "BNS Section 415",
        "title": "Organized Cyber Crime",
        "text": "Coordinated digital criminal activity by a group.",
        "simplified": "A group working together to commit cyber crimes."
    }
]

class LegalVectorStore:
    def __init__(self):
        print("Initializing Legal Engine (Lazy Mode)...")
        self.model = None
        self.index = None
        self.metadata = BNS_DATA
        self._indexed = False
        self.dimension = None

    def _initialize_model_and_index(self):
        if self._indexed:
            return

        print("Loading AI Model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Preparing Documents...")
        documents = [
            item["text"] + " " + item["simplified"]
            for item in self.metadata
        ]

        print("Generating Embeddings...")
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self.dimension = embeddings.shape[1]

        print("Building FAISS Index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype("float32"))

        self._indexed = True
        print("Legal Knowledge Base Ready.")

    def search(self, query: str, k=1):
        if not self._indexed:
            self._initialize_model_and_index()

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_vector.astype("float32"),
            k
        )

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results


# Initialize the brain
legal_engine = LegalVectorStore()
