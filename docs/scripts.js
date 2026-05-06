// Interactive swing phase preview for the GitHub Pages site.
const phaseData = [
    {
        label: "Address",
        description: "Establish posture alignment before initiating movement."
    },
    {
        label: "Takeaway",
        description: "Begin the swing with controlled shoulder and hip rotation."
    },
    {
        label: "Backswing",
        description: "Load power while maintaining a stable lead arm and spine angle."
    },
    {
        label: "Top",
        description: "Transition from backswing to downswing while keeping balance."
    },
    {
        label: "Downswing",
        description: "Accelerate the club through impact with coordinated rotation."
    },
    {
        label: "Impact",
        description: "Strike the ball with optimal clubface and body alignment."
    },
    {
        label: "Follow-through",
        description: "Finish the swing with full extension and controlled deceleration."
    }
];

const slider = document.querySelector("[data-phase-slider]");
const label = document.querySelector("[data-phase-label]");
const description = document.querySelector("[data-phase-description]");

const updatePhase = (index) => {
    if (!label || !description) {
        return;
    }

    const phase = phaseData[index] || phaseData[0];
    label.textContent = phase.label;
    description.textContent = phase.description;
};

if (slider) {
    updatePhase(Number(slider.value));
    slider.addEventListener("input", (event) => {
        updatePhase(Number(event.target.value));
    });
}
