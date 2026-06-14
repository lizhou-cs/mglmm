BASE_CAPTION_QUESTIONS = [
    'Could you please give me a description of the image?',
    'Can you provide a description of this image?',
    'Please provide a description of the image.',
    'Please describe the contents of the image.',
    'Could you give an explanation of what can be found within this picture?',
    'Could you give me an explanation of this picture?',
    'Could you provide me with an analysis of this photo?',
    'Can you give an explanation of this photo?',
    'Please provide an explanation of this picture.',
    'Could you provide me with an explanation of this photo?',
]

SHORT_CAPTION_PROMPTS = [
    "Please provide a brief response to the question.",
    "Please respond to the question concisely.",
    "Please provide a short format answer.",
    "Please keep your answer brief.",
    "Respond with a brief format.",
    "Answer with a short caption.",
]

LONG_CAPTION_PROMPTS = [
    "Please provide a detailed response to the question.",
    "Please provide a rich and detailed explanation.",
    "Respond with a comprehensive description.",
    "Give a thorough and detailed answer.",
    "Provide a long format answer.",
    "Answer with a detailed caption.",
    "Give a complete and detailed answer."
]

CAPTION_QUESTIONS = [
    'Could you please give me a detailed description of the image?',
    'Could you please give me a thorough description of the image?',
    'Can you provide a detailed description of this image?',
    'Can you provide a thorough description of the image?',
    'Please provide a detailed description of this image.',
    'Please provide a thorough description of the image.',
    'Please describe in detail the contents of the image.',
    'Could you give a comprehensive explanation of what can be found within this picture?',
    'Could you give me an elaborate explanation of this picture?',
    'Could you provide me with a detailed analysis of this photo?',
    'Can you give a comprehensive explanation of this photo?',
    'Please provide an elaborate explanation of this picture.',
    'Please provide a comprehensive explanation of this picture.',
    'Could you provide me with a detailed explanation of this photo?',
]

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

MULTIPLE_SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What are {class_name} in this image? Please respond with segmentation mask.",
    "What are {class_name} in this image? Please output segmentation mask.",
]

REGION_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

REGION_GROUP_QUESTIONS = [
    'Could you please give me a detailed description of these areas <region>?',
    'Can you provide a thorough description of the regions <region> in this image?',
    'Please describe in detail the contents of the boxed areas <region>.',
    'Could you give a comprehensive explanation of what can be found within <region> in the picture?',
    'Could you give me an elaborate explanation of the <region> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <region> in this photo?',
    'Help me understand the specific locations labeled <region> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <region> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <region> in this photo?',
    'What are the specific features of the areas marked <region> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <region> in this image?',
    'What can you tell me about the areas labeled <region> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <region> in this photo?',
    'I am interested in learning more about the regions marked <region> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <region> in this photo?',
    'What is the significance of the regions labeled <region> in this picture?',
    'I would like to know more about the specific locations designated <region> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <region> in this photo?',
    'What specific features can you tell me about the areas identified by <region> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <region> in this image?',
    'Can you provide a detailed account of the regions designated <region> in this photo?',
    'I am curious about the areas marked <region> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <region> in this image?',
    'Could you please provide a detailed description of the regions labeled <region> in this photo?',
    'What can you tell me about the features of the areas designated <region> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <region> in this image?',
    'I would like to know more about the specific locations identified by <region> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <region> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <region> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <region> in this photo?'
]

GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

GCG_LONG_QUESTIONS = [
    "Could you please give me a detailed description of the image? Please give a general overview of the image, followed by a detailed description and interleaved segmentation mask of each object visible within it.",
    "Can you provide a thorough description of this image? Start with an overall summary of the image's contents, proceed to explain each visible object in detail, and include the associated segmentation masks.",
    "Please describe in detail the contents of the image. Please first outline a general depiction of the image, then delve into specifics by describing each object and showcasing the segmentation masks related to those items",
    "Could you give a comprehensive explanation of what can be found within this picture? Please output an overall summary of the image's contents, followed by a detailed description at each object along with their segmentation masks.",
    "Could you give me an elaborate explanation of this picture? Please respond with a general overview of the image, followed by a detailed description and interleaved segmentation mask of each object visible within it.",
    "Could you provide me with a detailed analysis of this photo? Please give a general overview of the image, followed by a detailed description and interleaved segmentation mask of each object visible within it.",
]

GCG_SUB_QUESTIONS = [
    "Could you expand on the {label} in this image? Please offer a simple summary of the object, then give a clear description and the segmentation mask for each smaller object it contains.",
    "Can you describe the {label} further in this image? Please respond with an overview of the object, followed by a detailed description and interleaved segmentation mask for each child object within it.",
    "Could you give me a more detailed description of the {label}? Start with an overall summary of the object, proceed to ouput detailed descriptions and associated segmentation masks for its components",
    "Please describe the {label} further in this image. Please output an general overview of the object, followed by a detailed description and interleaved segmentation mask of each child object within it.",
    "Can you provide a more thorough description of the {label} in this image? Please offer a simple summary of the object, then give a clear description and the segmentation mask for each smaller object it contains.",
    "What more can you tell me about the {label} in the image? Please provide an initial outline of the object, continue with detailed descriptions and interleaved segmentation masks for its components.",
    "What further details can you give me about the {label} in the photo? Please output a simple summary of the object, then give a clear description and the segmentation mask for each smaller object it contains.",
    "I'd like more information on the {label} shown here. Please respond with an initial outline of the object, continue with detailed descriptions and interleaved segmentation masks for its components.",
]

GCG_SUB_ANSWERS = [
    "There are more details about its components.",
    "More information on its elements is available.",
    "There are further particulars on its parts.",
    "Additional details on its components are available.",
    "More information on its elements is provided.",
    "There are further particulars on its parts.",
    "There are further details concerning its child objects.",
    "More information on its child objects is available.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

SHORT_QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

LONG_QUESTION_LIST = [
    "{sent} Please respond with segmentation mask.",
    "{sent} Please output segmentation mask.",
    "{sent} Provide the segmentation mask.",
    "{sent} Output the segmentation mask.",
    "{sent} Please show the segmentation mask.",
    "{sent} I'd appreciate segmentation masks.",
    "{sent} Please highlight the segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

MR_SINGLE_ANSWER_LIST = [
    "{class_name} [SEG]."
]

MR_MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
]