# Gemini 프로젝트 로그

## 요약

이 문서는 Gemini 에이전트와 함께 로봇 탐사 시뮬레이션 프로젝트를 수정하고 개선한 작업 내역을 기록합니다.

## 진행 상황 (2025-09-19)

### 1. HOCBF 제어기 성능 개선을 위한 Nominal Control 고찰

- **배경**: HOCBF 기반 CBF-QP 제어기의 최종 성능은 Nominal Control Input (`u_ref`)의 품질에 크게 의존합니다. 현재 구현된 P-제어 기반의 Nominal Control은 목표 지점까지의 단순 거리와 각도만을 고려하므로, 동역학적 제약이나 장애물 정보를 반영하지 못하여 전반적인 제어 성능을 저하시킬 수 있습니다.

- **개선 아이디어**: 정교하고 적응적인 Nominal Control을 생성하기 위해 **CBF-Aware Safe Reinforcement Learning (RL)** 을 도입합니다.

- **Safe RL 도입의 장점**:
    - **학습 기반 최적화**: 경험을 통해 복잡한 제어 정책을 학습하여, 수동으로 설계하기 어려운 최적의 행동을 찾을 수 있습니다.
    - **안전성 보장 학습**: **Differentiable CBF**를 RL 프레임워크에 통합하여, 학습 과정에서부터 안전 제약조건을 만족하도록 정책을 유도합니다. 이를 통해 에이전트의 탐험(Exploration)과 최종 정책 모두 CBF에 의해 정의된 안전 영역 내에서 이루어지도록 보장할 수 있습니다.
    - **End-to-End 최적화**: 장기적인 보상을 최대화하면서 안전 제약도 만족하는 Nominal Control(`u_ref`)을 직접 출력하도록 학습하여, MPC와 CBF를 순차적으로 적용하는 방식보다 더 높은 성능을 기대할 수 있습니다.

- **다음 단계**:
    - CBF 제약조건을 미분 가능한(Differentiable) 형태로 공식화합니다.
    - Safe RL 에이전트를 위한 상태, 행동, 보상 함수를 설계합니다.
    - Differentiable CBF를 안전 레이어(Safety Layer) 또는 패널티(Penalty)로 통합한 학습 루프를 구현합니다.
    - 학습을 통해 Nominal Control을 생성하는 RL 정책을 훈련시킵니다.
    - RL 기반 Nominal Control의 성능을 기존 방식과 비교 분석합니다.
