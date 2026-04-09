using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// SortingAgent action handling (Step 09).
///
/// Handles both action-space variants from one codebase. The active mode is
/// detected at Initialize time by reading BehaviorParameters.ActionSpec:
///
///   Discrete [2]        → MA-POCA, PPO, DQN. Native framework masking via
///                         WriteDiscreteActionMask. Action 0 = do nothing,
///                         Action 1 = activate gate.
///
///   Continuous 1        → MADDPG. NO framework masking — a C# state guard
///                         in OnActionReceived prevents illegal activations
///                         during gate animation. Python masks the replay
///                         buffer separately.
///
/// Heuristic: Q activates Gate 0, W Gate 1, E Gate 2. Works in both modes.
/// Each agent only reacts to its own key (branchIndex → key).
/// </summary>
public partial class SortingAgent
{
    private enum ActionMode
    {
        Unknown,
        Discrete,    // Built-in trainers (MA-POCA, PPO, DQN)
        Continuous   // Custom trainers (MADDPG)
    }

    private ActionMode _actionMode = ActionMode.Unknown;
    private BehaviorParameters _behaviorParameters;

    /// <summary>
    /// Threshold above which a continuous gate action is treated as "activate".
    /// Applied ONLY in C#; the raw continuous value is what the Python critic sees.
    /// </summary>
    private const float ContinuousActivationThreshold = 0.5f;

    /// <summary>
    /// Called from SortingAgent.Initialize() to cache the action mode by
    /// inspecting the attached BehaviorParameters component.
    /// </summary>
    internal void DetectActionMode()
    {
        _behaviorParameters = GetComponent<BehaviorParameters>();
        Debug.Assert(_behaviorParameters != null,
            $"[SortingAgent {_branchIndex}] BehaviorParameters component missing.", this);

        if (_behaviorParameters == null)
        {
            _actionMode = ActionMode.Unknown;
            return;
        }

        ActionSpec spec = _behaviorParameters.BrainParameters.ActionSpec;

        if (spec.NumContinuousActions > 0)
        {
            _actionMode = ActionMode.Continuous;
            Debug.Assert(spec.NumContinuousActions >= 1,
                $"[SortingAgent {_branchIndex}] continuous mode requires >=1 continuous action, got {spec.NumContinuousActions}.");
        }
        else if (spec.BranchSizes != null && spec.BranchSizes.Length > 0)
        {
            _actionMode = ActionMode.Discrete;
            Debug.Assert(spec.BranchSizes[0] == 2,
                $"[SortingAgent {_branchIndex}] discrete mode expects branch 0 size 2, got {spec.BranchSizes[0]}.");
        }
        else
        {
            _actionMode = ActionMode.Unknown;
            Debug.LogError(
                $"[SortingAgent {_branchIndex}] BehaviorParameters has no actions configured.", this);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        switch (_actionMode)
        {
            case ActionMode.Discrete:
                {
                    // Framework has already applied WriteDiscreteActionMask below,
                    // so action 1 should only arrive when gate is Retracted. We
                    // still call TryActivateGate() which is idempotent — Activate()
                    // returns false if not actionable.
                    int gateAction = actions.DiscreteActions[0];
                    if (gateAction == 1)
                    {
                        TryActivateGate();
                    }
                    break;
                }

            case ActionMode.Continuous:
                {
                    // MADDPG: Python forces masked_action[0] = 0.0 when gate is
                    // not retracted, but we defend anyway in case of threshold
                    // drift or a raw inference path skipping the mask.
                    float gateAction = actions.ContinuousActions[0];
                    if (gateAction > ContinuousActivationThreshold
                        && _gate != null
                        && _gate.CurrentState == GateState.Retracted)
                    {
                        TryActivateGate();
                    }
                    break;
                }

            case ActionMode.Unknown:
            default:
                // DetectActionMode already logged an error. Fail silent here
                // so we don't spam the Console every decision step.
                break;
        }
    }

    /// <summary>
    /// Discrete action masking. Called by the ML-Agents framework every
    /// decision on discrete-mode agents. NOT called on continuous-mode agents.
    /// Masks action index 1 (activate) when the gate is not Retracted, so
    /// the policy cannot select an illegal activation during gate animation.
    /// </summary>
    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // Safety: if called on a continuous agent somehow, do nothing.
        if (_actionMode != ActionMode.Discrete) return;

        if (_gate != null && _gate.CurrentState != GateState.Retracted)
        {
            // Modern API: SetActionEnabled(branch, actionIndex, isEnabled).
            // Do NOT use the old SetMask() method.
            actionMask.SetActionEnabled(0, 1, false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Branch index → key: 0=Q, 1=W, 2=E.
        KeyCode myKey;
        switch (_branchIndex)
        {
            case 0: myKey = KeyCode.Q; break;
            case 1: myKey = KeyCode.W; break;
            case 2: myKey = KeyCode.E; break;
            default: myKey = KeyCode.None; break;
        }

        // Use GetKey (held) not GetKeyDown so a held key survives across
        // multiple decision intervals. Repeat activations during cooldown
        // are harmless — DiverterGate.Activate() rejects them.
        bool pressed = (myKey != KeyCode.None) && Input.GetKey(myKey);

        if (_actionMode == ActionMode.Discrete)
        {
            ActionSegment<int> discreteOut = actionsOut.DiscreteActions;
            if (discreteOut.Length > 0)
            {
                discreteOut[0] = pressed ? 1 : 0;
            }
        }
        else if (_actionMode == ActionMode.Continuous)
        {
            ActionSegment<float> continuousOut = actionsOut.ContinuousActions;
            if (continuousOut.Length > 0)
            {
                continuousOut[0] = pressed ? 1f : 0f;
            }
        }
        // Unknown mode: leave buffer at default (zeros).
    }

    /// <summary>
    /// Single-point gate activation. Ignores the return value — unsuccessful
    /// activations (during animation or cooldown) are expected and harmless;
    /// the policy learns to time them.
    /// </summary>
    private void TryActivateGate()
    {
        if (_gate == null) return;
        _gate.Activate();
    }
}