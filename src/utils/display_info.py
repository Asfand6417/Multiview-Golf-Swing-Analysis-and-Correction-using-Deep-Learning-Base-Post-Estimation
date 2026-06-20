import cv2

def display_info(img, fps, mpjpe=None, phase=None, additional_text=None):
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if mpjpe is not None:
        cv2.putText(img, f'MPJPE: {mpjpe:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if phase:
        cv2.putText(img, f'Phase: {phase}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if additional_text:
        for i, text in enumerate(additional_text):
            cv2.putText(img, text, (10, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img